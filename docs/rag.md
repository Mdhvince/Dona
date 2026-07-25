# RAG : ingestion, index, retrieval

Pipeline documentaire de l'assistant : les fichiers des deux Google Drive
sont transcrits, découpés, embeddés dans Chroma, et interrogés par une
recherche hybride. Tout tourne en local via Ollama.

```
  Google Drive ──> transcription VLM ──> validation ──> chunking ──> embeddings ──> Chroma
  (PDF, images,      (qwen3.5:9b)      anti-hallu     structurel   (qwen3-emb:8b)
   txt, md)                                            Markdown
```

## Ingestion (`src/ingest.py`)

### Transcription par modèle vision

Chaque page PDF est rendue en image (pypdfium2, 150 DPI) puis transcrite en
Markdown par `qwen3.5:9b` (`reasoning = false` : la transcription n'a pas
besoin de raisonner, ~30s/page). Raison de ce choix : les extracteurs texte
et les parsers de layout dissocient les libellés de leurs valeurs sur les
documents administratifs en colonnes (avis d'imposition...) ; seul un modèle
vision préserve l'association. Les images (.png/.jpg) passent par le même
VLM : transcription du texte visible + courte description. Les prompts de
transcription sont dans `src/prompt.py`.

### Validation anti-hallucination (`validate_transcription`)

Tout nombre présent dans la transcription doit exister dans la couche texte
du PDF (pypdf), qui contient les vraies valeurs même mal placées.
Comparaison à deux granularités (brute et chiffres recollés "9 570" ->
"9570") plus un test par sous-chaîne pour les identifiants longs. Un nombre
introuvable déclenche un avertissement (console + UI via `warnings` du
résultat de sync) ; un PDF scanné sans couche texte est signalé comme non
vérifiable.

### Chunking structurel (`src/document_processing.py`)

La transcription étant du Markdown, le découpage suit la structure :
`MarkdownHeaderTextSplitter` sur les titres (une section reste entière,
tables comprises), puis `RecursiveCharacterTextSplitter` uniquement sur les
sections dépassant `chunk_size`. Le chemin de titres est préfixé au texte du
chunk (contexte pour l'embedding) et stocké en métadonnée `section`.

### Ingestion incrémentale (`sync`)

Chaque chunk porte `source` (chemin) et `mtime`. `sync()` compare le disque
et l'index : nouveau fichier -> indexé ; fichier modifié (tolérance 1s, les
montages cloud font varier les mtimes) -> ré-indexé ; fichier disparu ->
chunks supprimés ; le reste n'est pas retouché. Une reconstruction complète
est un `sync()` sur collection vide (`--full`), nécessaire après un
changement de modèle d'embedding (espaces vectoriels incompatibles).

### Périmètre

- Racines : les deux montages Google Drive (`DOCS_DIRS`).
- Formats : .pdf, .png, .jpg, .jpeg, .txt, .md.
- Tout chemin traversant un dossier commençant par `_` est exclu (privé).
- Métadonnées par chunk : `source`, `page`, `mtime`, `section`, et un tag
  par niveau de dossier (`tag_1="05 - Clients"`), filtrables dans Chroma.

## Embeddings et base vectorielle

`qwen3-embedding:8b` via l'API OpenAI-compatible d'Ollama, persistance
Chroma dans `vectordb/`. L'ingestion et la requête partagent
`config.embedding_client` : les deux doivent utiliser exactement le même
modèle.

## Retrieval (`src/retrieval.py`)

Recherche hybride fusionnée par Reciprocal Rank Fusion :

- **Sémantique** : similarité Chroma sur les embeddings.
- **Keyword** : BM25 sur une tokenisation française (minuscules, accents
  retirés, stopwords exclus, racinisation Snowball).
- **RRF** : score 1/(rrf_k + rang) additionné entre les classements.

Le pipeline est une fonction déterministe, sans LLM : `search(queries)`
accepte plusieurs formulations et fusionne chaque couple (requête, méthode)
comme un classement de plus. Paramètres dans `config.toml [retriever]`.
BM25 est un index en mémoire, reconstruit au démarrage de la webapp et
après chaque ré-indexation.

## Commandes

```bash
uv run python -m src.ingest                         # ingestion incrémentale
caffeinate -i uv run python -m src.ingest --full    # reconstruction complète (~30s/page)
```

## Limites connues

- Pas d'évaluation systématique (jeu de questions/réponses de référence).
- La validation numérique ne couvre ni les PDF scannés sans couche texte ni
  les erreurs de transcription non numériques.
- Un document signalé par la validation mérite un contrôle visuel.
