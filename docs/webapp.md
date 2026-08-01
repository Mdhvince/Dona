# Webapp

Interface Flask (`webapp/`, convention flask.md : `run.py` à la racine,
package avec `views.py`, `templates/`, `static/`). Port 5001 (le 5000 est
occupé par le récepteur AirPlay de macOS). Thème sombre à deux panneaux
(sources à gauche, conversation à droite), sans framework CSS. Le pipeline
(agent, retriever, outils MCP) est construit une fois au démarrage et
reconstruit après chaque ré-indexation.

## Routes

- `GET /` : page unique - panneau Sources à gauche (repliable), fil de
  conversation à droite avec le composeur en bas.
- `POST /ask` (`{question, thread_id}`) : flux NDJSON sur
  `agent.stream(stream_mode=["messages", "values"])` :
  - battements `{phase, tokens}` throttlés issus du flux de tokens :
    réflexion en cours avec compteur, préparation d'appel d'outil,
    rédaction de la réponse (le contenu du raisonnement n'est jamais envoyé
    au navigateur, seul son avancement) ;
  - événements d'étape `{tool, args}` et `{retrieved}` issus du flux
    d'états ;
  - ligne finale : réponse HTML (Markdown converti côté serveur, extension
    tables, assaini par nh3 - python-markdown laisse passer le HTML brut,
    vecteur XSS via un document indexé malveillant), `status` ("ok" ou
    "error"), sources citées [{name, page, url}], nombre de documents
    consultés, requêtes émises, `failed_tools`, et `models` - les noms des
    modèles ayant généré les réponses du tour, lus dans les
    `response_metadata` posés par le framework (jamais déclarés par le
    modèle lui-même) et affichés en texte discret sous la réponse. `status` vaut "error"
    quand tous les appels d'outils du tour ont échoué (le front affiche un
    message d'erreur fixe à la place de la réponse du modèle) et "partial"
    quand une partie a réussi (la réponse s'affiche sous un bandeau fixe
    listant les outils en échec) : un échec d'accès n'est jamais présenté
    comme une donnée ("agenda vide"...), et la détection vient du code
    (sentinelle de la couche outils), pas du modèle.
- `POST /confirm` (`{thread_id, approved}`) : reprend un run interrompu sur
  un outil à effet de bord et poursuit le même flux NDJSON. Le flux de
  `/ask` s'arrête sur une ligne `{confirm: [{tool, args}]}` quand une
  approbation est requise ; le front affiche alors une carte listant les
  arguments réels de l'appel, avec Confirmer / Annuler (toute autre issue
  vaut refus).
- `GET /source?path=...` : sert un document indexé dans le navigateur
  (les liens file:// sont bloqués sur une page HTTP), restreint aux racines
  indexées (403 sinon).
- `POST /reindex` : ingestion incrémentale dans un thread d'arrière-plan,
  un seul run à la fois. La reconstruction complète est réservée au
  terminal (batch de plusieurs heures, voir docs/rag.md).
- `GET /reindex/status` : `{running, result, error}`, pollé par le front
  pendant une ré-indexation (la progression détaillée reste dans le terminal
  qui fait tourner le serveur).

## Comportement du front

- **Fil de conversation** : les échanges s'empilent à l'écran (question en
  bulle alignée à droite, réponse en texte pleine largeur). Un message
  d'accueil et trois questions d'amorce cliquables ouvrent chaque nouvelle
  conversation. Le `thread_id` vit dans le localStorage ; "Nouvelle
  conversation" (menu ⋮) en génère un neuf et vide l'écran. Un rechargement
  de page vide l'affichage mais pas la mémoire du fil, conservée côté
  serveur.
- **Radar d'activité** : pendant l'attente, une carte affiche un radar à 8
  axes correspondant aux capacités de l'agent (Réflexion, Documents, Agenda
  pro, Agenda perso, Recherche web, Qonto, Mail pro, Mail perso - les axes
  non encore branchés restent plats). La courbe se déforme vers l'axe de ce
  que l'agent fait réellement : les phases du flux de tokens pointent
  Réflexion, chaque appel d'outil pointe son ou ses axes via `TOOL_ROUTES`
  (motifs ancrés, le plus spécifique d'abord ; un outil inconnu retombe sur
  Réflexion). Un axe déjà utilisé garde une valeur intermédiaire, la forme
  précédente reste en rémanence le temps du morph. L'en-tête donne la phrase
  d'état et le temps écoulé, la ligne mono sous le radar le détail courant
  (requêtes de l'outil, compteur de tokens, extraits récupérés) et la barre
  du bas une progression asymptotique (le nombre d'étapes étant inconnu,
  elle n'atteint jamais 100%). La carte disparaît quand la réponse arrive.
- **Panneau Sources** : les sources citées s'accumulent en cartes
  (nom de fichier + page) sur toute la conversation, dédupliquées ; un clic
  ouvre le document dans un nouvel onglet à la page citée (`#page=N`). Le
  compteur est repris dans le composeur, et le panneau se replie via son
  bouton d'en-tête. Un état vide explicite s'affiche tant qu'aucune source
  n'est citée.
- **Échecs d'outils** : `status: "error"` remplace la réponse par un message
  fixe ; `status: "partial"` insère un bandeau d'avertissement listant les
  outils en échec au-dessus de la réponse.
- **Menu ⋮** : "Nouvelle conversation" et "Re-indexer" ; l'état d'indexation s'affiche dans le header et les fichiers en échec dans un
  panneau dépliable sous celui-ci.

## Commandes

```bash
uv run python run.py    # http://127.0.0.1:5001
uv run pytest           # tests unitaires
```
