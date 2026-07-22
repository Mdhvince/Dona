# Architecture RAG de l'assistant

Assistant personnel de question/réponse sur les documents des Google Drive
(Myelink + perso), 100% local : aucun document ni aucune requête ne quitte la
machine. Stack : Ollama (LLM, VLM, embeddings), Chroma (base vectorielle),
LangChain (orchestration), Flask (interface).

```
                    INGESTION (batch, incrémentale)
  Google Drive ──> transcription VLM ──> validation ──> chunking ──> embeddings ──> Chroma
  (PDF, images,      (qwen3.5:9b)      anti-hallu     structurel   (qwen3-emb:8b)
   txt, md)                                            Markdown

                    REQUETE (webapp Flask)
  question ──> MultiQuery ──> recherche hybride ──> LLM (gpt-oss:20b) ──> réponse
               (variantes)    BM25 + sémantique      sortie structurée     + source
                              fusion RRF             (Pydantic)
```

## Ingestion (`src/ingest.py`)

### Parsing par modèle vision, pas par extracteur texte

Choix central du projet : chaque page PDF est rendue en image (pypdfium2,
150 DPI) puis transcrite en Markdown par un modèle vision local
(`qwen3.5:9b` via Ollama, `reasoning=false` car la transcription n'a pas
besoin de raisonner, ~12x plus rapide, ~30s/page).

Pourquoi pas une extraction classique ? Testé sur un avis d'imposition
(formulaire en colonnes, texte positionné) :

- pypdf (défaut) : libellés et montants dissociés, montants en vrac en fin de page ;
- pypdf `layout` : correct sur ce document, mais fragile en général ;
- Docling (modèles de layout IBM) : échoue aussi, chunks de montants orphelins ;
- VLM : transcription fidèle, chaque montant en face de son libellé.

Le VLM est plus lent et non déterministe, mais l'ingestion est un batch rare
et sa faiblesse (inventer des valeurs) est précisément ce que la validation
sait détecter (voir ci-dessous). Les images (.png/.jpg) passent par le même
VLM : transcription du texte visible + courte description.

### Validation anti-hallucination (`validate_transcription`)

Tout nombre présent dans la transcription VLM doit exister dans la couche
texte du PDF (pypdf) : elle contient les vraies valeurs, même quand
l'extraction les place mal. Comparaison à deux granularités (brute et
chiffres recollés "9 570" -> "9570") plus un test par sous-chaîne pour les
identifiants longs segmentés différemment. Un nombre introuvable déclenche
un avertissement console listant les valeurs suspectes ; un PDF scanné sans
couche texte est signalé comme non vérifiable. Ce garde-fou a détecté en test
un chiffre perdu par le VLM dans un identifiant fiscal de 18 chiffres.

### Chunking structurel (`src/document_processing.py`)

La transcription étant du Markdown, le découpage suit la structure :
`MarkdownHeaderTextSplitter` sur les titres (une section reste entière,
tables comprises), puis `RecursiveCharacterTextSplitter` uniquement sur les
sections dépassant `chunk_size` (1000). Le chemin de titres ("Avis d'impôt
2024 > CALCUL DU SOLDE") est préfixé au texte du chunk (contexte pour
l'embedding) et stocké en métadonnée `section`.

### Ingestion incrémentale (`sync`)

Chaque chunk porte en métadonnées `source` (chemin) et `mtime` (date de
modification du fichier). `sync()` compare le disque et l'index :

- fichier absent de l'index -> transcrit et indexé ;
- fichier modifié (`mtime` plus récent) -> anciens chunks supprimés, ré-indexé ;
- fichier disparu du disque -> chunks supprimés ;
- le reste n'est pas retouché (pas de re-transcription inutile).

Une reconstruction complète = `sync()` sur une collection vide
(`--full` en CLI, bouton "Tout re-indexer" dans la webapp). Obligatoire
après un changement de modèle d'embedding : les espaces vectoriels ne sont
pas compatibles.

### Périmètre indexé

- Racines : les deux montages Google Drive (`DOCS_DIRS`).
- Formats : .pdf, .png, .jpg, .jpeg, .txt, .md.
- Dossiers privés : tout chemin traversant un dossier commençant par `_`
  est exclu.
- Métadonnées par chunk : `source`, `page`, `mtime`, `section`, et un tag
  par niveau de dossier (`tag_1="05 - Clients"`, `tag_2="Techplaces"`),
  filtrables dans Chroma.

## Embeddings et base vectorielle

`qwen3-embedding:8b` via l'API OpenAI-compatible d'Ollama (choisi pour la
qualité multilingue et la confidentialité : les documents ne partent plus
chez un fournisseur d'API). Persistance Chroma dans `vectordb/`.
L'ingestion et la requête partagent le même client (`config.embedding_client`) :
les deux doivent utiliser exactement le même modèle.

## Retrieval (`src/retrieval.py`)

Recherche hybride fusionnée par Reciprocal Rank Fusion :

- **Sémantique** : similarité Chroma sur les embeddings.
- **Keyword** : BM25 sur une tokenisation adaptée au français (minuscules,
  accents retirés, stopwords français exclus, racinisation Snowball) pour
  que "numero" matche "numéro".
- **RRF** : score 1/(rrf_k + rang) additionné entre les deux classements.

Par-dessus, `MultiQueryRetriever` (LangChain) génère des variantes de la
question avec le LLM pour élargir le rappel. Paramètres dans
`config.toml [retriever]`. BM25 est un index en mémoire reconstruit au
démarrage de la webapp et après chaque ré-indexation.

## Génération (`src/response_helper.py`)

Sortie structurée Pydantic via `with_structured_output` :

- `response` : la réponse en Markdown, sans mention de source ;
- `source` : le numéro `[i]` de l'extrait utilisé, pas une URI : un entier
  est difficile à halluciner et se valide par bornes.

Le code reconstruit ensuite le lien réel depuis les métadonnées du chunk
cité (`source_readable` pour le terminal, `source_path`/`source_page` pour
la webapp). Prompt système et descriptions de champs en français, comme le
corpus.

## Webapp (`webapp/`, convention flask.md)

- `GET /` : page unique (question, réponse, panneau Sources).
- `POST /ask` : question -> réponse HTML (Markdown converti côté serveur)
  + source {name, page, url}.
- `GET /source?path=...` : sert le document (les liens file:// sont bloqués
  en HTTP), restreint aux racines indexées. Le panneau Sources affiche un
  aperçu du PDF (iframe, ouvert à la page citée) cliquable.
- `POST /reindex` : ingestion incrémentale en thread d'arrière-plan, un seul
  run à la fois (bouton "Re-indexer"). La reconstruction complète est
  volontairement absente de l'interface : c'est un batch de plusieurs heures,
  réservé au terminal (voir Commandes).
- `GET /reindex/status` : `{running, done, total, current, result, error}`,
  pollé par le front pour afficher la progression (x/total, %).

Pas d'historique : chaque question efface l'interaction précédente.

## Commandes

```bash
uv run python src/ingest.py          # ingestion incrémentale (CLI)
uv run python run.py                 # webapp sur http://127.0.0.1:5000
```

Reconstruction complète de l'index (uniquement en terminal) - nécessaire
après un changement de modèle d'embedding ou de prompt de transcription,
~30s/page soit 2h30-3h pour tout le corpus. `caffeinate -i` empêche la mise
en veille pendant le batch ; Ollama doit être démarré :

```bash
caffeinate -i uv run python src/ingest.py --full
```

## Limites connues et pistes

- Pas encore d'évaluation systématique (jeu de questions/réponses de
  référence à rejouer après chaque changement d'ingestion) : prévu.
- La validation numérique ne couvre pas les PDF scannés (pas de couche
  texte) ni les erreurs de transcription non numériques.
- La transcription VLM d'un document signalé mérite un contrôle visuel
  ponctuel (alerte console pendant l'ingestion).
- Ré-indexation depuis la webapp : le serveur doit rester lancé pendant
  toute la durée du run (pas de reprise en cours de route).
