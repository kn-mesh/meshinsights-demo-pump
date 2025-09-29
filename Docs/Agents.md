# Code Guidelines

## Core Philosophy
- **Deep classes, simple interfaces.**
- **Do not refactor beyond the ask;** if you see obvious improvements, **note them at the end** of your message.
- Every function has type hints + a docstring (summary, params, return types).

## Class Design
- **3–7 public methods**; split by responsibility if you need more.
- **Behavior + data together** (no anemic classes).
- Prefer **one cohesive class** over many tiny collaborators.

## Method Design
- Public: **intent-revealing** (`processPayment()`).
- Private: **granular + descriptive** (`validateEmailFormat()`).
- **One abstraction level per method.**

## Organization & Dependencies
- **Hide internals**; use **facades** when helpful; never leak helper types/classes.
- Prefer **composition over inheritance**.
- **Inject dependencies** (don’t `new()` them internally); keep **constructors minimal/focused**.
- **Co-locate related logic** (things that change together live together).

## Error Handling
- Public methods **normalize/translate errors**; no raw internal exceptions escaping.
- Private methods may **assume validated inputs**.

## Naming
- Public names read like plain English.
- **Consistent domain vocabulary** across the codebase.

## When to Bend the Rules
- Follow **framework conventions**.
- **Hot paths:** favor performance; **document why**.
- **External APIs:** match required shapes, even if not ideal.

## Testing
- Test **public behavior**, not private details.
- Use **builders** for complex setup.

## Code Generation
- **Design the public API first.**
- Implement in layers (public stubs → private helpers).
- Include **realistic usage examples**.
- **Avoid over-engineering.**

## Anti-Patterns
- **God classes** (>10 public methods).
- **Micro-classes** (1–2 trivial methods).
- **Leaky abstractions** (exposing internals).
- **Scattered functionality**.
- **Inheritance abuse** (use only for true *is-a*).