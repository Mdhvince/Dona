# Agent

Agent LangGraph (`create_agent`, LangChain 1.x) construit dans
`src/agent.py`, modèle de chat configuré dans `config.toml [llm]` (Ollama,
API native). Le prompt système et les descriptions d'outils vivent dans
`src/prompt.py` (en français, comme le corpus et les réponses). Persona :
tutoiement, ton direct, règles numérotées (chercher d'abord, extraits =
données, vérification du titulaire, fidélité des montants, priorité au
document le plus récent, date du jour injectée dynamiquement à chaque appel
via un middleware `dynamic_prompt`).

## Outils

- **`rag_medhys_files`** (`src/tools.py`, fabrique `make_rag_tool`) : le
  retriever hybride est capturé par closure, le schéma expose uniquement
  `queries: list[str]`. `response_format="content_and_artifact"` : les
  extraits numérotés (nom de fichier + page) partent dans le contexte de
  l'agent, les métadonnées `{path, page}` voyagent en artifact dans le
  `ToolMessage` sans repasser par le LLM.
- **`calendar_find_event`** (`make_calendar_finder`) : outil composite qui
  encode la stratégie de recherche d'événement en code (recherche fullText
  par terme discriminant sur chaque compte, repli sur un listing large) et
  rend les candidats dédupliqués au modèle, qui n'a plus qu'à reconnaître
  le bon (les intitulés d'événements sont souvent abrégés).
- **Serveurs MCP** (`load_mcp_tools`) : déclarés en blocs `[[mcp]]` dans
  config.toml, connectés au démarrage via langchain-mcp-adapters (stdio),
  outils renommés `<serveur>_<outil>`. Whitelist d'outils par serveur (lecture
  seule ; un nom absent du serveur déclenche un avertissement) ; un serveur
  injoignable est ignoré avec un avertissement, l'agent continue sans lui.
  Les outils MCP étant async, ils sont enveloppés en outils synchrones (une
  boucle asyncio par appel).

## Détection des échecs d'outils

La véracité des échecs est garantie par le code, jamais par le modèle. La
couche outils émet la sentinelle `TOOL_ERROR` quand un appel échoue :
exception de l'enveloppe, payload d'erreur détecté par le finder, ou
résultat non-JSON d'un serveur déclaré `json_result = true` (les serveurs
Google renvoient leurs erreurs en texte brut dans un résultat normal, un
payload valide est toujours du JSON). `tool_failures` scanne les messages
du tour et associe chaque échec au nom de l'outil ; le statut structuré de
la réponse devient "error" (tous les appels ont échoué) ou "partial"
(certains ont réussi), et l'UI affiche un message fixe ou un bandeau
d'avertissement listant les outils en échec - le modèle ne narre jamais
un échec à l'utilisateur.

## Calendrier

Calendar passe par le serveur communautaire `@cocal/google-calendar-mcp`
(stdio, un process npx par compte). `GOOGLE_ACCOUNT_MODE` sélectionne le
token du compte (créé via `npx @cocal/google-calendar-mcp auth <compte>`,
stocké dans `~/.config/google-calendar-mcp`, client OAuth "Desktop" dans
`~/.secrets/google-oauth.json`) et `fixed_args` épingle le paramètre
`account` de chaque outil : le modèle ne peut pas interroger un autre
compte que celui de l'instance. `json_result = true` déclare le contrat de
payload pour la détection d'erreurs. Les outils d'écriture (create/update/
delete-event) sont hors whitelist en attendant un mécanisme de confirmation
human-in-the-loop.

## Conversation

L'agent reçoit un checkpointer `SqliteSaver` (`conversations.db`, gardé
hors de l'agent pour survivre aux reconstructions post-ré-indexation ; les
conversations survivent aux redémarrages du serveur). Chaque requête porte
un `thread_id` : même thread = même conversation, avec résolution des
références ("et son adresse ?"). `current_turn()` isole les messages du
tour courant, le checkpointer renvoyant tout l'historique.

## Citations

La sortie structurée est découplée de la boucle d'agent : une grammaire de
sortie imposée pendant la boucle entre en conflit avec le tool calling
selon les modèles. L'agent répond librement, puis `extract_citations` fait
un second appel court et structuré (sans outils) : le modèle coche, parmi
les documents réellement récupérés (artifacts), ceux que la réponse
utilise - schéma strict (nom de fichier, page). `validate_citations` écarte
toute citation ne correspondant pas à un document récupéré ; le chemin
affiché vient toujours de l'artifact, jamais du LLM. Aucun document
récupéré = pas d'appel d'extraction.
