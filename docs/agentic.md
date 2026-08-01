# Agent

## Modèles et posture de confidentialité

Deux fournisseurs de modèles, choisis par domaine (`config.toml`, un bloc
par rôle, clé `provider`) :

- **Local (Ollama)** : l'agent des documents et calendriers
  (`[agent_local]`, qwen avec réflexion) et le small talk
  (`[conversation]`, mêmes poids sans réflexion). Les documents personnels
  ne quittent jamais la machine.
- **Melious (API OpenAI-compatible, hébergement RGPD de modèles open
  source)** : le routeur (`[router]`, gpt-oss-120b, `reasoning_effort =
  "low"`) et l'agent critique (`[agent_critical]`, kimi-k2.6) - les
  domaines où une erreur d'appel d'outil coûte plus cher que la latence
  (banque, emails à venir), confiés à un modèle plus fort.

Décision assumée : chaque question transite par Melious (routage), et les
données bancaires y sont traitées (agent critique). Le RAG, les documents
et le small talk restent locaux. Clé API : `MELIOUS_DONA_API_KEY` dans
`.env`. En cas de panne ou d'absence de crédits Melious, le routeur se
replie sur la branche locale : l'assistant continue de fonctionner hors
ligne, en mode dégradé (sans les outils bancaires).

## Routage

Un graphe LangGraph place un routeur devant trois branches partageant le
même historique de messages :

- **conversation** : small talk, réponse directe locale sans réflexion.
- **agent_local** : documents (RAG) et calendriers, sur le modèle local.
- **agent_critical** : banque (Qonto) et emails à venir, plus le RAG (les
  questions transverses "facture <-> transaction" se composent ici), sur le
  modèle fort.

Le routeur répond par CONVERSATION, LOCAL ou CRITIQUE, avec les derniers
échanges en contexte (texte seul, tronqué, jamais les traces d'outils : voir
`router_history`) pour résoudre les questions de suite ("et le mois
d'avant ?"). **Tout sauf un CONVERSATION ou un LOCAL explicite mène à
l'agent critique** - doute, verdict illisible, domaines mélangés : l'agent
critique sait tout faire (il a aussi le RAG), l'inverse est faux. Seule la
panne du routeur se replie sur l'agent local, pour que l'assistant survive
hors ligne.

Chaque branche agent est un `create_agent` (LangChain 1.x) construit dans
`src/agent.py` à partir d'un `AgentProfile` (modèle, outils, prompt
système, outils à confirmation). Les prompts vivent dans `src/prompt.py`
(en français, comme le corpus et les réponses), composés de blocs partagés
(identité, extraits = données, vérification du titulaire, fidélité des
montants, citations) plus les règles d'outils propres à chaque agent : le
prompt local ne mentionne pas Qonto, le prompt critique ne mentionne pas
les calendriers. La date du jour est injectée à chaque appel via un
middleware `dynamic_prompt`.

## Outils

- **`rag_medhys_files`** (`src/tools.py`, fabrique `build_rag_tool`) : le
  retriever hybride est capturé par closure, le schéma expose uniquement
  `queries: list[str]`. `response_format="content_and_artifact"` : les
  extraits numérotés (nom de fichier + page) partent dans le contexte de
  l'agent, les métadonnées `{path, page}` voyagent en artifact dans le
  `ToolMessage` sans repasser par le LLM.
- **`calendar_find_event`** (`build_calendar_finder`) : outil composite qui
  encode la stratégie de recherche d'événement en code (recherche fullText
  par terme discriminant sur chaque compte, repli sur un listing large) et
  rend les candidats dédupliqués au modèle, qui n'a plus qu'à reconnaître
  le bon (les intitulés d'événements sont souvent abrégés).
- **Serveurs MCP** (`load_mcp_tools`) : déclarés en blocs `[[mcp]]` dans
  config.toml, connectés au démarrage via langchain-mcp-adapters (stdio),
  outils renommés `<serveur>_<outil>`. La clé `agent` du bloc affecte le
  serveur à une branche (`local` par défaut, `critical` pour Qonto) ;
  `mcp_tools_of_agent` partitionne les outils chargés par ce critère, et
  seul le RAG est partagé entre les deux agents. Whitelist d'outils par
  serveur (un nom absent du serveur déclenche un avertissement) ; un serveur
  injoignable est ignoré avec un avertissement, l'agent continue sans lui.
  Les outils MCP étant async, ils sont enveloppés en outils synchrones (une
  boucle asyncio par appel).

## Détection des échecs d'outils

La véracité des échecs est garantie par le code, jamais par le modèle. La
couche outils émet la sentinelle `TOOL_ERROR` quand un appel échoue :
exception de l'enveloppe, payload d'erreur détecté par le finder, ou
résultat non-JSON d'un serveur déclaré `json_result = true` (les serveurs
Google renvoient leurs erreurs en texte brut dans un résultat normal, un
payload valide est toujours du JSON). `tool_outcomes` scanne les messages
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

Le serveur est affecté à l'agent critique (`agent = "critical"`). Il expose
62 outils ; la whitelist en retient 59 : les 31 lectures s'exécutent
librement, les 28 écritures (factures, devis, cartes, demandes de virement,
clients...) sont toutes déclarées en `confirm` et passent par la
confirmation explicite. Les 3 outils `delete_*` (client, facture, devis)
sont exclus de la whitelist elle-même : l'irréversible n'est pas accessible
à l'agent, même confirmé. Côté Qonto, aucune écriture sensible n'aboutit
sans SCA (2FA) dans leur app : `create_multi_transfer_request` ne crée
qu'une demande de virement en attente, jamais un virement exécuté.

## Actions à effet de bord

Les outils déclarés en `confirm` dans un bloc `[[mcp]]` (`create-event`
sur les deux agendas, les 28 écritures Qonto) ne s'exécutent jamais seuls :
le `HumanInTheLoopMiddleware` de leur agent interrompt le graphe avant
l'appel, l'état interrompu est persisté par le checkpointer, et la reprise
se fait sur approbation explicite (`POST /confirm`). La carte de
confirmation affiche les **arguments réels** de l'appel, jamais la
paraphrase du modèle : ce qui est approuvé est ce qui sera envoyé. Toute
décision autre qu'une approbation explicite est un refus, l'agent en est
informé et poursuit sa réponse. Sur les agendas, update/delete-event
restent hors whitelist.

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
