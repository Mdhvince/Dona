# Webapp

Interface Flask (`webapp/`, convention flask.md : `run.py` à la racine,
package avec `views.py`, `templates/`, `static/`). Port 5001 (le 5000 est
occupé par le récepteur AirPlay de macOS). Look minimaliste noir/blanc/gris,
sans framework CSS. Le pipeline (agent, retriever, outils MCP) est construit
une fois au démarrage et reconstruit après chaque ré-indexation.

## Routes

- `GET /` : page unique - champ question, réponse, sources.
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
    consultés, requêtes émises, et `failed_tools`. `status` vaut "error"
    quand tous les appels d'outils du tour ont échoué (le front affiche un
    message d'erreur fixe à la place de la réponse du modèle) et "partial"
    quand une partie a réussi (la réponse s'affiche sous un bandeau fixe
    listant les outils en échec) : un échec d'accès n'est jamais présenté
    comme une donnée ("agenda vide"...), et la détection vient du code
    (sentinelle de la couche outils), pas du modèle.
- `GET /source?path=...` : sert un document indexé dans le navigateur
  (les liens file:// sont bloqués sur une page HTTP), restreint aux racines
  indexées (403 sinon).
- `POST /reindex` : ingestion incrémentale dans un thread d'arrière-plan,
  un seul run à la fois. La reconstruction complète est réservée au
  terminal (batch de plusieurs heures, voir docs/rag.md).
- `GET /reindex/status` : `{running, done, total, current, result, error}`,
  pollé par le front pendant une ré-indexation.

## Comportement du front

- Le fil d'activité (à droite du spinner) montre le travail de l'agent en
  direct : une ligne vivante pour les phases, des lignes empilées pour les
  appels d'outils (`[Calling <outil>]: <args>`) et les extraits récupérés.
  Chaque étape tient sur une ligne (troncature ellipse) : 1 ligne = 1 appel
  d'outil ou 1 phase.
- Sous la réponse, les sources en liste compacte muted ("Sources :
  avis_2024.pdf, p.2 · ...", plus le nombre de documents consultés) ; un
  clic ouvre le document dans un nouvel onglet à la page citée (`#page=N`).
- Les requêtes de recherche de l'agent s'affichent en muted sous le champ
  question une fois la réponse rendue.
- Conversation : le `thread_id` vit dans le localStorage (survit aux
  rechargements de page) ; le bouton "Nouvelle conversation" en génère un
  neuf et vide l'écran. L'affichage ne montre qu'un échange à la fois, mais
  la mémoire du fil persiste côté serveur.
- Après une ré-indexation, les alertes de validation s'affichent dans un
  panneau dépliable sous le header.

## Commandes

```bash
uv run python run.py    # http://127.0.0.1:5001
uv run pytest           # tests unitaires
```
