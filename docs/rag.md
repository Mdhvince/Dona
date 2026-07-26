# RAG : ingestion, index, retrieval

Pipeline documentaire de l'assistant : les fichiers des deux Google Drive
sont transcrits, découpés, embeddés dans Chroma, et interrogés par une
recherche hybride. Tout tourne en local via Ollama.

```
  Google Drive ──> conversion ──> chunking ──> embeddings ──> Chroma
  (PDF, images,    (glm-ocr,     par format     (bge-m3)
   docx, pptx,     MarkItDown)
   html, txt, md)
```

## Ingestion (`src/ingest.py`)

### Transcription par modèle vision

Chaque page PDF est rendue en image (pypdfium2, 150 DPI) puis transcrite en
Markdown par `glm-ocr` (`reasoning = false` : la transcription n'a pas
besoin de raisonner). Raison de ce choix : les extracteurs texte
et les parsers de layout dissocient les libellés de leurs valeurs sur les
documents administratifs en colonnes (avis d'imposition...) ; seul un modèle
vision préserve l'association. Les images (.png/.jpg) passent par le même
VLM : transcription du texte visible + courte description. Les prompts de
transcription sont dans `src/prompt.py`.

### Formats balisés (`ingest_markup`)

Les .docx, .pptx et .html portent déjà leur structure dans du balisage
(OOXML, HTML). MarkItDown la lit et la rend en Markdown : styles de titres
et arbre DOM sont repris tels quels, aucune structure n'est inventée. Ni
VLM ni heuristique sur ces formats.

### Chunking par format

`load_file()` renvoie le format qu'il a produit, et `chunk_documents()`
aiguille vers le découpeur correspondant. La stratégie suit le format de
**sortie** du chargeur, jamais le suffixe d'origine : un PDF transcrit par
le VLM est du Markdown, et se découpe comme tel.

- **markdown** (PDF, images, .md, .docx, .pptx, .html) :
  `MarkdownHeaderTextSplitter` sur les titres (une section reste entière,
  tables comprises), puis `RecursiveCharacterTextSplitter` uniquement sur
  les sections dépassant `chunk_size`. Le chemin de titres est préfixé au
  texte du chunk (contexte pour l'embedding) et stocké en métadonnée
  `section`.
- **text** (.txt) : aucune structure à suivre, découpage récursif seul, sur
  les frontières de paragraphe puis de phrase puis de mot. Pas de métadonnée
  `section`.

Convertir un .txt en Markdown reviendrait à inventer des titres absents de
la source, qui se retrouveraient préfixés dans l'embedding et dans le
contexte lu par l'agent : le repli récursif est explicite, pas subi.

### Ingestion incrémentale (`Ingestor.run`)

Chaque chunk porte `source` (chemin) et `mtime`. `run()` compare le disque
et l'index : nouveau fichier -> indexé ; fichier modifié (tolérance 1s, les
montages cloud font varier les mtimes) -> ré-indexé ; fichier disparu ->
chunks supprimés ; le reste n'est pas retouché. Une reconstruction complète
est un `run()` sur collection vide (`--full`), nécessaire après un
changement de modèle d'embedding (espaces vectoriels incompatibles).

### Périmètre

- Racines : les montages Google Drive listés dans `DOCS_DIRS` (variable
  `.env`, voir `.env.example` : chemins personnels hors du repo ; comme les
  emails et le chemin des credentials OAuth, interpolés `${VAR}` dans
  `config.toml` au chargement).
- Formats : .pdf, .png, .jpg, .jpeg, .md, .txt, .docx, .pptx, .html, .htm.
  Les formats tabulaires (.csv, .xlsx) ne sont pas pris en charge : ce sont
  des données structurées, pas du document.
- Tout chemin traversant un dossier commençant par `_` est exclu (privé).
- Métadonnées par chunk : `source`, `page`, `mtime`, `section`.

## Embeddings et base vectorielle

`bge-m3` via l'API OpenAI-compatible d'Ollama, persistance
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
- Aucune vérification automatique des transcriptions : une erreur du modèle
  vision se propage silencieusement dans l'index.
