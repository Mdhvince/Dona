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

                    REQUETE (webapp Flask, agent LangGraph)
  question ──> agent (gpt-oss:20b) ──> tool rag_medhys_files ──> réponse
               décide quand chercher    recherche hybride         + sources
               et formule les requêtes  BM25 + sémantique, RRF    (artifacts)
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

Le pipeline hybride est une fonction déterministe, volontairement sans LLM :
`search(queries)` accepte plusieurs formulations et fusionne chaque couple
(requête, méthode) comme un classement RRF de plus. Paramètres dans
`config.toml [retriever]` (k=6 après fusion). BM25 est un index en mémoire
reconstruit au démarrage de la webapp et après chaque ré-indexation.

## Agent (`src/agent.py`, `src/tools.py`)

Le RAG est un outil d'un agent LangGraph (`create_agent`, LangChain 1.x) :

- **`rag_medhys_files`** (`src/tools.py`) : fabrique `make_rag_tool(retriever)`
  qui capture le retriever par closure ; le schéma expose uniquement
  `queries: list[str]`. La docstring est le prompt de l'outil : périmètre du
  corpus et consigne de requêtes explicites et nommées ("passeport de Medhy
  Vinceslas", jamais "mon passeport") - c'est elle qui résout les possessifs,
  l'ancienne étape de réécriture de requête est devenue inutile. L'outil est déclaré
  `response_format="content_and_artifact"` : le texte des extraits
  (numérotés, nom de fichier + page, via `format_context`) part dans le
  contexte de l'agent, les métadonnées `{path, page}` voyagent en artifact
  dans le `ToolMessage` sans repasser par le LLM.
- **L'agent** (`build_agent`) décide s'il faut chercher, avec quelles
  requêtes, et peut enchaîner plusieurs recherches (multi-hop) si les
  premiers extraits ne suffisent pas. Politique "chercher d'abord" : toute
  question factuelle passe par le RAG, et si rien n'est trouvé l'assistant
  le dit au lieu de répondre de sa connaissance générale. Le prompt système
  reprend les règles historiques : extraits = données (anti-injection),
  vérification du titulaire, fidélité absolue des montants et identifiants,
  priorité au document le plus récent avec année précisée, date du jour
  injectée.
- **Outils MCP** (`tools.load_mcp_tools`) : les serveurs déclarés en
  `[[mcp]]` dans config.toml (stdio) sont connectés au démarrage via
  langchain-mcp-adapters et leurs outils rejoignent l'agent, renommés
  `<serveur>_<outil>` - le même serveur peut tourner une fois par compte
  (calendar_pro, calendar_perso, chacun son token OAuth). Whitelist
  d'outils en lecture seule par serveur (un nom absent du serveur déclenche
  un avertissement au démarrage) ; un serveur injoignable est ignoré avec
  un avertissement, l'agent continue sans lui. Les outils MCP étant async,
  ils sont enveloppés en outils synchrones (une boucle asyncio par appel)
  pour rester compatibles avec la stack Flask/agent synchrone. `fixed_args`
  épingle par instance les paramètres que le modèle ne doit pas contrôler :
  le compte d'un serveur multi-comptes, typiquement - la valeur configurée
  écrase toujours celle du modèle (vérifié en réel : le modèle invente des
  noms de compte). Connectés : calendar_pro / calendar_perso (Google
  Calendar, lecture seule, OAuth local dans ~/.config/google-calendar-mcp) ;
  les actions d'écriture (create/update/delete-event) sont volontairement
  hors whitelist en attendant le human-in-the-loop.
- **Conversation** : l'agent reçoit un checkpointer `SqliteSaver`
  (`conversations.db` à la racine, gardé hors de l'agent pour survivre aux
  ré-indexations ; les conversations survivent aussi aux redémarrages du
  serveur) et chaque requête porte un `thread_id` : même thread = même
  conversation, avec résolution des références ("et son adresse ?").
  `current_turn()` isole les messages du tour courant, car le checkpointer
  renvoie tout l'historique (le fil d'activité et les compteurs ne
  concernent que le tour en cours ; les citations restent validées contre
  tout le thread).
- **Citations** : découplées de la boucle d'agent, par choix d'architecture
  model-agnostic. Imposer un schéma de sortie pendant la boucle entre en
  conflit avec le tool calling selon les modèles (`ProviderStrategy` :
  grammaire qui empêche qwen3 d'appeler les outils ; `ToolStrategy` : outil
  final que gpt-oss ignore). L'agent répond donc librement, puis
  `extract_citations` fait un second appel court et structuré (sans outils,
  donc sans conflit) : le modèle coche, parmi les documents réellement
  récupérés (artifacts des `ToolMessage`), ceux que la réponse utilise -
  schéma strict (nom de fichier, page). `validate_citations` écarte toute
  citation ne correspondant pas à un document récupéré ; le chemin affiché
  vient toujours de l'artifact, jamais du LLM. Aucun document récupéré
  (agenda, conversation) = pas d'appel d'extraction. L'UI montre les sources
  citées puis le nombre de documents consultés ; `collect_queries` expose
  les requêtes envoyées. La réponse HTML est assainie par `nh3` avant envoi
  au front (python-markdown laisse passer le HTML brut : vecteur XSS via un
  document indexé malveillant).

## Webapp (`webapp/`, convention flask.md)

- `GET /` : page unique (question, réponse, sources).
- `POST /ask` (`{question, thread_id}`) : flux NDJSON sur
  `agent.stream(stream_mode=["messages", "values"])`. Le `thread_id` vit
  dans le localStorage du navigateur ; le bouton "Nouvelle conversation" en
  génère un neuf et vide l'écran. Pendant l'attente, le fil d'activité (à
  droite du spinner) combine deux granularités : des battements `{phase,
  tokens}` throttlés issus du flux de tokens (réflexion en cours avec
  compteur, préparation d'appel d'outil, rédaction de la réponse - le
  contenu du raisonnement n'est jamais envoyé au navigateur, seul son
  avancement), et des événements d'étape `{tool, args}` ("[Calling ...]")
  et `{retrieved}` issus du flux d'états. Une ligne finale porte la réponse HTML (Markdown
  converti et assaini côté serveur), les sources citées [{name, page, url}]
  et le nombre de documents consultés.
- `GET /source?path=...` : sert le document (les liens file:// sont bloqués
  en HTTP), restreint aux racines indexées. Les sources s'affichent en
  liste compacte muted sous la réponse ("Sources : avis_2024.pdf, p.2 ...") ;
  un clic ouvre le document dans un nouvel onglet, à la page citée.
- `POST /reindex` : ingestion incrémentale en thread d'arrière-plan, un seul
  run à la fois (bouton "Re-indexer"). La reconstruction complète est
  volontairement absente de l'interface : c'est un batch de plusieurs heures,
  réservé au terminal (voir Commandes).
- `GET /reindex/status` : `{running, done, total, current, result, error}`,
  pollé par le front pour afficher la progression (x/total, %). Les alertes
  de la validation anti-hallucination remontent dans `result.warnings` et
  s'affichent dans un panneau dépliable.

Pas d'historique : chaque question efface l'interaction précédente.

## Commandes

```bash
uv run python -m src.ingest          # ingestion incrémentale (CLI)
uv run python run.py                 # webapp sur http://127.0.0.1:5000
uv run pytest                        # tests unitaires (tests/)
```

Reconstruction complète de l'index (uniquement en terminal) - nécessaire
après un changement de modèle d'embedding ou de prompt de transcription,
~30s/page soit 2h30-3h pour tout le corpus. `caffeinate -i` empêche la mise
en veille pendant le batch ; Ollama doit être démarré :

```bash
caffeinate -i uv run python -m src.ingest --full
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
