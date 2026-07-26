# Agent

## Routage

Un graphe LangGraph place un routeur devant deux branches partageant le même
historique de messages. Le routeur et la branche conversation utilisent le
même modèle que l'agent mais avec `reasoning = false` (section `[router]`) :
`think` étant un paramètre par requête, les poids déjà chargés servent les
trois rôles, sans mémoire supplémentaire. Une salutation est ainsi traitée
en une fraction de seconde au lieu des ~20 s que coûte la réflexion.

Le routeur ne répond que par CONVERSATION ou OUTILS, et **tout sauf un
CONVERSATION explicite mène à l'agent complet** - réponse illisible, doute,
panne du routeur : une vraie question traitée sans outils serait
hallucinée, alors qu'une salutation traitée par l'agent ne coûte que du
temps. La branche conversation reçoit l'historique du fil (nettoyé de ses
appels d'outils) pour comprendre les références, mais aucun outil.

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
compte que celui de l'instance. `default_args` comble les paramètres que le
serveur exige mais que son schéma présente comme optionnels - `calendarId`
et la fenêtre temporelle (`timeMin`/`timeMax`, écrits `@now-30d` /
`@now+365d` et résolus à l'appel) : sans eux, un appel sans fenêtre est
rejeté par le serveur. `json_result = true` déclare le contrat de payload
pour la détection d'erreurs. Les outils d'écriture (create/update/
delete-event) sont hors whitelist en attendant un mécanisme de confirmation
human-in-the-loop.

## Banque (Qonto)

Serveur MCP officiel de Qonto (`https://mcp.qonto.com/mcp`, transport
http), authentifié par le flow OAuth du protocole MCP - découverte des
métadonnées, enregistrement dynamique du client, PKCE et rafraîchissement,
fournis par `OAuthClientProvider` du SDK et branchés via
`src/mcp_auth.py` (stockage sur disque dans `~/.secrets`, un fichier par
serveur) :

```bash
uv run python -m src.mcp_auth qonto   # autorisation unique, liste les outils
```

Le serveur expose 62 outils dont beaucoup d'écriture (cartes, virements,
factures) ; la whitelist n'en retient que 7, tous en lecture : solde et
organisation, transactions, relevés, factures clients et fournisseurs,
cartes. C'est autant une mesure de sécurité qu'un choix de qualité - un
modèle local choisit mal parmi 62 outils.

## Actions à effet de bord

Les outils déclarés en `confirm` dans un bloc `[[mcp]]` (aujourd'hui
`create-event` sur les deux agendas) ne s'exécutent jamais seuls : le
`HumanInTheLoopMiddleware` interrompt le graphe avant l'appel, l'état
interrompu est persisté par le checkpointer, et la reprise se fait sur
approbation explicite (`POST /confirm`). La carte de confirmation affiche
les **arguments réels** de l'appel, jamais la paraphrase du modèle : ce qui
est approuvé est ce qui sera envoyé. Toute décision autre qu'une
approbation explicite est un refus, l'agent en est informé et poursuit sa
réponse. Les autres écritures (modification, suppression) restent hors
whitelist.

## Fraîcheur des recherches

Un middleware (`fresh_retrieval`) réécrit l'historique avant chaque appel
modèle : les tours précédents ne gardent que leur conversation (questions et
réponses), leurs appels d'outils et extraits sont retirés, le tour en cours
reste intact. Sans lui, le modèle voit les extraits déjà récupérés, estime
son contexte suffisant et ne relance pas de recherche - jusqu'à répondre
"je n'ai pas trouvé" sur un document qu'il n'a jamais cherché. Les appels
d'outils passés sont retirés avec leurs résultats (un appel orphelin est
rejeté par certains fournisseurs). Chaque question déclenche donc sa propre
recherche, et le fil reste compréhensible via le texte des réponses.

## Conversation

L'agent reçoit un checkpointer `SqliteSaver` (`conversations.db`, gardé
hors de l'agent pour survivre aux reconstructions post-ré-indexation ; les
conversations survivent aux redémarrages du serveur). Chaque requête porte
un `thread_id` : même thread = même conversation, avec résolution des
références ("et son adresse ?"). `current_turn()` isole les messages du
tour courant, le checkpointer renvoyant tout l'historique.

## Citations

Citation en ligne, résolue en code. Chaque extrait rendu par l'outil RAG
porte un marqueur unique (`[3f2a]`, id aléatoire : les marqueurs doivent
rester uniques entre les plusieurs appels d'outil d'un même tour), présent
aussi dans l'artifact du `ToolMessage`. Le prompt demande à l'agent de
recopier le marqueur juste après l'information qui en vient ;
`parse_citations` les retrouve dans la réponse, les résout en sources via
les artifacts et les retire du texte affiché. Entièrement déterministe : le
modèle qui cite est celui qui a lu les extraits, et un marqueur ne
correspondant à aucun extrait récupéré est écarté.

Aucune grammaire de sortie n'est imposée pendant la boucle d'agent : elle
entre en conflit avec le tool calling selon les modèles. Les sources
affichées portent un label unique (nom de fichier, plus le dossier parent
quand plusieurs fichiers partagent ce nom - les documents comptables se
répètent d'un exercice à l'autre).
