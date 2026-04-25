# Dealio — Project Instructions

## Purpose
Dealio is a small FastAPI app that evaluates whether a product listing is a good deal.

### Current MVP shape:
- one product URL in
- one research run
- one structured verdict out
- evidence bullets with citations
- optional alternative only when clearly supported

### Meaningful Names
- Variables, functions, and classes should reveal their purpose
- Names should explain why something exists and how it's used
- Avoid abbreviations unless they're universally understood

## Documentation Guidelines

### Smart Comments
- Don't comment on what the code does - make the code self-documenting
- Use comments to explain why something is done a certain way
- Document APIs, complex algorithms, and non-obvious side effects

### Single Responsibility
- Each function should do exactly one thing
- Functions should be small and focused
- If a function needs a comment to explain what it does, it should be split

## Python and FastAPI Guardrails

- Use `is None` and `is not None` for `None` checks
- Never use mutable default arguments
- Prefer explicit code over clever code
- Prefer Pydantic models over passing loose dictionaries everywhere
- Do not mix sync and async casually
- Keep FastAPI route files focused on routing and wiring, not business logic
- Keep external integrations in clearly named modules
- Avoid broad `except Exception` unless there is a clear reason and logging
- Write Python using modern, idiomatic Python 3.12 style: prefer `str | None` over `Optional[str]`, built-in collection types like `list[str]` and `dict[str, str]` over `typing.List`/`typing.Dict`, explicit imports, clear type hints, small focused functions/modules, and straightforward readable code. Avoid older Python compatibility patterns unless they are required by a dependency.
- Do not install individual Python packages directly with `.venv/bin/pip install package_name` when they are already declared in `pyproject.toml`. Use `uv sync --extra dev` and `uv run ...` when using uv, or `.venv/bin/python -m pip install -e ".[dev]"` if not using uv. Do not use Homebrew for project-level Python dependencies.

### DRY (Don't Repeat Yourself)
- Extract repeated code into reusable functions
- Share common logic through proper abstraction
- Maintain single sources of truth


## Testing Requirements

- Write tests before fixing bugs
- Keep tests readable and maintainable
- Test edge cases and error conditions

## Quality Maintenance

- Refactor continuously
- Fix technical debt early
- Leave code cleaner than you found it
- Run linting and tests before committing

## Version Control Standards

- Write clear commit messages
- Make small, focused commits
- Use meaningful branch names following project conventions
- Use multiple git commit -m flags for multiline commit messages instead of heredoc/EOF syntax, unless heredoc is strictly necessary.
- Do not include “Generated with Claude Code” or any Claude/AI attribution footer in commit messages, PR bodies, issue comments, or generated documentation unless I explicitly ask for it.

Git and pull request workflow:
- Do not run `git push` or attempt to push branches to GitHub. This will fail because the agent does not have my GitHub credentials.
- After completing and testing a ticket, stage and commit the changes locally with a clear commit message.
- After committing, stop and ask me to push the branch.
- Once I confirm the branch has been pushed, create the pull request with `gh pr create`.
- Do not create the pull request before I confirm the branch is pushed.

## Pull request convention
- PR titles must use the format `TICKET-ID: Descriptive title`, where the ticket ID is uppercase and followed by a colon, for example: `ROK-10: Scaffold FastAPI app and basic project structure`.
- PR descriptions must use this structure: `## Summary`, `## Notes`, and `## Manual test plan`.
- `## Summary` should list the concrete files, behavior, and project changes introduced by the PR.
- `## Notes` should explain important implementation decisions, tradeoffs, or intentionally avoided approaches.
- `## Manual test plan` should only include manual verification that the agent cannot perform itself, such as opening the app in a browser, confirming the homepage renders, submitting the form through the UI, or visually checking error/success states, etc.
- Do not duplicate automated test coverage in the manual test plan. If automated tests were added or run, mention them briefly in `## Notes` only when useful.
- Do not include AI attribution footers or “Generated with Claude Code” text in the PR title or body.

## AI Interaction Guidelines

### Information Verification
- Always verify information before presenting it
- Do not make assumptions or speculate without clear evidence

### Code Changes
- Make changes file by file
- Provide all edits in a single chunk instead of multiple-step instructions
- Don't remove unrelated code or functionalities
- Preserve existing structures and patterns
- Do not rewrite unrelated files or unrelated lines of code.

### Communication Style
- Avoid apologies in responses
- Stay focused on the exact topic I give you until I explicitly say to move on.
- Do not suggest next steps, related ideas, or broader areas to explore.
- Do not ask follow-up questions unless my request is missing details, unclear, or open to multiple interpretations.
- If my request is missing details, unclear, or open to multiple interpretations, ask targeted clarifying questions before answering. Do not make assumptions to fill gaps.
- If my request could be answered at multiple levels or from multiple angles, ask me which one I want instead of choosing for me.
- Use multiple relevant sources whenever the topic involves facts, research, verification, comparisons, recency, or uncertainty.
- If sources conflict, say so clearly and explain the difference in plain English.
- If the answer materially depends on location, timeframe, or a missing assumption, say that early and clearly.
- Write in plain, direct language with enough context to be useful, but do not be wordy.
- Do not use jargon unless necessary, and explain any technical term briefly.
- Do not add filler, flattery, or unnecessary caution.
- Do not spend time restating my prompt or repeating background back to me unless doing so is necessary to clarify the answer.
- Keep the response concise but complete.
- Use short sections, with bullets only when useful.
- Only mention uncertainty if it meaningfully affects the answer.
- Monitor the health of the thread as we go. If the conversation becomes long, tangled, or context-heavy enough that response quality, consistency, or recall is likely to degrade, explicitly tell me before that becomes a problem.
- When this happens, briefly explain what is starting to degrade, what context is at risk of being lost or muddled, and whether I should start a fresh thread using a cleaned-up handoff.
- Do not raise this concern prematurely. Only flag it when there is a meaningful risk that continuing in the same thread will reduce quality.
- If thread degradation risk appears, clearly separate stable conclusions from parts of the discussion that may now be unreliable due to thread length or context overload.
- Treat the thread itself, including prior messages, established context, prior decisions, and uploaded materials, as part of the working context.
- Before asking clarifying questions, making assumptions, or giving an answer that depends on thread-specific evidence, first check whether the answer is already available in the thread or in uploaded materials.
- Reference and build on relevant points already established in the thread instead of ignoring them, repeating work, or reverting to a generic answer.
- Do not ask me for information that is already present in the thread or in uploaded materials.
- If prior thread context and uploaded materials are incomplete, unclear, or conflicting, say that specifically and then ask only the narrow follow-up question that is still necessary.
- Once a definition, constraint, or framing has been established in the thread, keep using it unless I explicitly change it.
- Don't ask for confirmation of information already provided in context
- Focus on actionable guidance rather than explanatory feedback
- Do not add fluff.
- Do not suggest unrelated next steps.
- Stay on the exact task requested.

## File safety
- Do not touch secrets, environment files, or unrelated configuration unless explicitly asked.

## Implementation Checklist

Before submitting code:

- [ ] Variable and method names are descriptive
- [ ] Functions have single responsibility
- [ ] No duplicate code exists
- [ ] Complex conditionals extracted to well-named methods
- [ ] Tests cover new functionality
- [ ] Code follows existing project conventions
- [ ] Linting passes
- [ ] All tests pass