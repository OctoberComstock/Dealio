# Dealio

Dealio is a small FastAPI app that checks whether a product listing looks like a good deal.
Paste in a product URL, and Dealio extracts the listing details, researches comparable offers,
validates the evidence, and returns a structured verdict with citations.

The current version is an MVP portfolio demo, not a consumer production app. It is built to show
practical agent architecture, evidence-grounded research, and safety checks around live web tools.

## What it does

Dealio takes one product URL and runs one research pass:

1. Fetch the submitted product page.
2. Extract the product name, merchant, and listed price.
3. Search for current comparable listings.
4. Fetch useful result pages for more evidence.
5. Validate comparable offers for product identity, size, availability, and purchasability.
6. Return a verdict: `Good Deal`, `Fair`, `Overpriced`, or `Insufficient Data`.
7. Include evidence bullets with source URLs and, when clearly supported, a cheaper alternative.

The goal is not to scrape every store on the internet. The goal is to make a focused, explainable
deal judgment from the evidence available during that run.

## Why I built this

Dealio is part of my agentic-systems portfolio work. I built it to get hands-on experience with
practical AI agent architecture: tool use, live web research, evidence grounding, safety checks,
evals, and deployment.

I also used the project to start building production intuition in Python and FastAPI after spending
most of my career in Ruby on Rails. The project was built with AI-assisted development: I used
coding agents to accelerate implementation and help me learn these new tech stacks and systems while driving the product direction,architecture
decisions, PR review, debugging, testing strategy, and eval design.

## Key features

- Live product URL analysis
- Fresh web research through Tavily
- Product extraction from static metadata, visible page text, and rendered pages
- Evidence-linked verdicts
- Alternative recommendation logic
- Availability and sold-out filtering
- Purchasable-offer validation
- Prompt-injection resistance for fetched page content
- URL normalization and tracking-parameter cleanup
- DNS and private-network safety checks before page fetches
- Model-in-the-loop eval harness with mocked search and fetch fixtures
- Password-gated public demo to control API usage

## How the agent works

Dealio keeps route handling, persistence, tool code, and agent orchestration separated. A submitted
URL moves through this flow:

```text
FastAPI/Jinja UI
   |
   v
Product extraction
   |
   v
Research agent + tools
   |
   v
Candidate validation
   |
   v
Verdict + evidence + alternative
   |
   v
SQLite result storage
```

The agent can search the web, fetch pages, and submit a final verdict. Search results and fetched
pages are treated as untrusted evidence, not instructions. Before rendering the final answer,
Dealio checks that evidence URLs and alternative URLs came from observed tool results.

Alternative recommendations are intentionally conservative. Dealio only includes an alternative
when it is clearly supported by observed evidence and passes checks for same product or size,
availability, purchasability, and meaningful savings.

## Tests and evals

Dealio uses both deterministic tests and model-in-the-loop evals:

- Unit and integration tests cover deterministic behavior such as config loading, URL
  normalization, page fetching, product identity, schema validation, database storage, and route
  behavior.
- Evals run the real research agent against controlled mocked search and fetch fixtures. This keeps
  the judgment-heavy behavior repeatable while still testing how the model responds to realistic
  evidence.

Current eval cases cover:

- Clear good-deal, fair, and overpriced verdicts
- Best purchasable alternative selection
- Avoiding sold-out listings
- Rejecting cheaper listings for the wrong size or variant
- Returning `insufficient_data` when evidence is weak
- Prompt-injection resistance
- Citation grounding to observed URLs

## Safety and cost controls

Dealio includes several guardrails because it works with user-submitted URLs and paid APIs:

- URL validation before accepting a submission
- URL normalization to remove common tracking parameters
- DNS resolution checks before fetching pages
- Private-network and loopback address blocking for fetched URLs
- Per-run search and fetch budgets
- Agent timeout handling
- Evidence grounding checks before results are shown
- Password-gated demo access to avoid unlimited public token and API usage
- `.env` and local database files excluded from Git

No real API keys, passwords, session secrets, Railway secret values, local `.env` contents, or
database contents should be committed to this repository.

## Tech stack

- Python 3.12
- FastAPI
- Jinja templates
- Anthropic / Claude
- Tavily
- Playwright
- SQLite
- Docker
- Railway
- uv
- pytest
- Ruff

## Running locally

Install dependencies:

```bash
uv sync --extra dev
```

Create a local environment file:

```bash
cp .env.example .env
```

Fill in local values using placeholders like these:

```text
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ANTHROPIC_MODEL=claude-sonnet-4-6
TAVILY_API_KEY=your_tavily_api_key_here
DEALIO_DEMO_PASSWORD=choose_a_demo_password
SESSION_SECRET_KEY=generate_a_random_secret
DATABASE_PATH=./dealio.db
LOG_LEVEL=DEBUG
```

Do not commit `.env` or real secret values.

Install the browser runtime used by Playwright:

```bash
uv run playwright install chromium
```

Start the app:

```bash
uv run fastapi dev app/main.py
```

Then open `http://127.0.0.1:8000`.

## Tests

Run the automated test suite:

```bash
uv run pytest
```

Run linting:

```bash
uv run ruff check .
```

Run the eval harness:

```bash
uv run python evals/run_dealio_evals.py
```

The eval harness requires `ANTHROPIC_API_KEY`. It does not require `TAVILY_API_KEY` because search
and fetch results are mocked from fixtures.

## Deployment

Dealio is set up for Dockerfile-based deployment on Railway. The Docker image installs uv, project
dependencies, and Chromium through Playwright so rendered product pages can be fetched in
production.

Railway provides `PORT` automatically. Runtime configuration should be set through Railway
environment variables, using placeholders like:

```text
ANTHROPIC_API_KEY=your_anthropic_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
DEALIO_DEMO_PASSWORD=choose_a_demo_password
SESSION_SECRET_KEY=generate_a_random_secret
DATABASE_PATH=/data/dealio.db
LOG_LEVEL=INFO
```

The live demo is password-gated to control API usage. Demo access can be provided separately
without publishing the password in the repository.

## Status and limitations

- MVP portfolio demo using SQLite, not a consumer production app
- One submitted URL, one research run, one structured verdict
- Live pricing depends on publicly available product pages and search results
- Some retailers block automation, require JavaScript, or show bot checks
- Results depend on the quality and freshness of available evidence
- Alternative recommendations are omitted unless the evidence clearly supports them
