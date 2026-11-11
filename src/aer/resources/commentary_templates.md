# Commentary Templates
_Python fills {placeholders}. No LLM call when a full template match exists._
_Last updated: YYYY-MM-DD_

---

## How Templates Are Used

1. `utils/conflict_classifier.py` classifies the conflict → returns template key
2. `utils/template_filler.py` selects the template by key, fills `{placeholders}` from agent JSON
3. If all placeholders resolve → return filled string directly, no LLM call
4. If any placeholder cannot resolve from agent JSON → fall through to LLM with template key injected as context

Placeholder values always come from the structured agent output dict. No inference, no LLM.

---

## Market Overview Templates

### TMPL_MARKET_OVERVIEW
```
{macro_regime_sentence}. {risk_sentence}. {valuation_sentence}. {main_caveat}.
```

Placeholder sources:
- `macro_regime_sentence`: derived from `macro.regime` + `MARKET_CONTEXT.md` regime description
- `risk_sentence`: derived from `risk.score` + `kb/econ/risk_framework.md` bucket
- `valuation_sentence`: derived from `fundamental.rationale.valuation`
- `main_caveat`: derived from top-severity conflict pattern label, if any

---

## Conflict Templates

### TMPL_MACRO_OVERRIDE
```
The {agent_id} agent signals {view} with {confidence_pct}% confidence, but the current
macro regime ({regime}) structurally weakens near-term directional calls of this type.
{regime_implication}. This configuration is consistent with a bear-market rally dynamic
rather than a durable directional move.
```

Placeholder sources:
- `agent_id`, `view`, `confidence_pct`: from conflicting agent's output dict
- `regime`: from `macro.regime`
- `regime_implication`: from `kb/econ/macro_regimes.md` — the one-sentence implication for this regime

---

### TMPL_MACRO_OVERRIDE_RISK
```
The {agent_id} agent signals {view} with {confidence_pct}% confidence. In the current
{regime} regime, structural deterioration is the dominant force: near-term directional calls
are systematically compromised by credit stress, liquidity withdrawal, or earnings compression.
{regime_implication}. Derisking discipline takes precedence over signal-following here — reduce
exposure rather than seeking confirmation of the bullish case.
```

Placeholder sources:
- `agent_id`, `view`, `confidence_pct`: from conflicting agent's output dict
- `regime`: from `macro.regime`
- `regime_implication`: from `kb/econ/macro_regimes.md` — the one-sentence implication for this regime

---

### TMPL_VALUATION_MOMENTUM
```
Fundamental valuation ({valuation_label}) and technical momentum ({technical_signal})
are pointing in opposite directions on {ticker}. This pattern is consistent with
late-cycle momentum continuation, where price diverges from intrinsic value before
mean-reversion. Short-term horizon favors the technical signal; medium-term horizon
favors the fundamental signal. Horizon alignment is the primary decision variable here.
```

Placeholder sources:
- `valuation_label`: from `fundamental.rationale.valuation`
- `technical_signal`: from `technical.view`
- `ticker`: from `context.ticker`

---

### TMPL_VALUATION_MOMENTUM_VOLATILE
```
Fundamental valuation ({valuation_label}) and technical momentum ({technical_signal})
are diverging on {ticker} in an elevated-VIX environment. In volatile regimes, momentum
signals are subject to sharp reversals and valuation anchors become unstable reference
points. Signal weighting should reflect regime uncertainty: neither the fundamental nor
the technical view should be treated as high-conviction until regime stabilizes. Reduce
position sizing and widen exit criteria accordingly.
```

Placeholder sources:
- `valuation_label`: from `fundamental.rationale.valuation`
- `technical_signal`: from `technical.view`
- `ticker`: from `context.ticker`

---

### TMPL_SENTIMENT_DIVERGENCE_RISK_ON
```
Strong negative sentiment (score: {sentiment_score}) against a bullish {other_agent}
signal on {ticker} in a risk-on macro environment may indicate a short-term overreaction.
Historical base rate for sentiment-driven pullbacks in risk-on regimes reverting within
{horizon} days is elevated. Monitor for sentiment stabilization before increasing conviction.
```

### TMPL_SENTIMENT_DIVERGENCE_RISK_OFF
```
Negative sentiment (score: {sentiment_score}) in a {regime} macro environment may be
a leading indicator of deterioration not yet captured in {other_agent} signals for {ticker}.
Weight the sentiment signal more heavily in stress regimes — it tends to lead price by
{horizon} days in historical risk-off periods.
```

---

### TMPL_SMART_MONEY_ALIGNED
```
Insider buying ({insider_summary}) and institutional call flow ({call_flow_summary})
are aligned with a bullish fundamental case for {ticker}. This is a high-conviction
configuration. Note: execution risk and current liquidity conditions ({liquidity_note})
remain relevant even under strong signal alignment.
```

### TMPL_SMART_MONEY_UNCONFIRMED
```
Insider buying ({insider_summary}) and call flow ({call_flow_summary}) are present
for {ticker}, but fundamental analysis does not confirm the bullish case ({valuation_label}).
This configuration is consistent with an information asymmetry play or speculative positioning.
Elevated caution warranted; position sizing should reflect the fundamental uncertainty.
```

---

### TMPL_SPECULATIVE_FLOW
```
Elevated call flow ({call_flow_summary}) for {ticker} is not accompanied by insider
confirmation. This is consistent with speculative retail or short-term institutional
positioning rather than informed directional conviction. Gamma risk at {nearest_expiry}
expiry and volatility crush post-event are material concerns.
```

---

### TMPL_PELOSI_CONFLICT
```
The Congress trading signal ({pelosi_view}) conflicts with fundamental valuation
({valuation_label}) for {ticker}. Where this pattern occurs, the most common
explanation is legislative information asymmetry on regulatory or procurement outcomes.
This is not a signal to follow without acknowledging the legal, ethical, and
information-reliability constraints on Congress trading data.
```

---

### TMPL_HIGH_CONVICTION_RISK
```
{n_aligned} agents are aligned on {view} for {ticker} (risk score: {risk_score}/10).
High cross-agent conviction does not neutralize elevated portfolio risk. A risk score
above {RISK_SCORE_WARNING_THRESHOLD} requires explicit position sizing discipline and
defined exit criteria regardless of signal alignment. Do not size this position as if
conviction reduces downside exposure.
```

Placeholder sources:
- `RISK_SCORE_WARNING_THRESHOLD`: from `utils/constants.py`

---

### TMPL_REGIME_CONFLICT
```
The macro regime classifier ({macro_regime}) and the technical signal ({technical_signal})
are implying inconsistent market conditions for {ticker}. A {technical_signal} technical
signal in a macro-classified {macro_regime} regime has an elevated historical false-positive
rate. Resolve the regime disagreement before acting on either signal.
```

---

## Model Explanation Templates

### TMPL_MODEL_EXPLANATION
```
The {model_name} model uses {key_features} to estimate {target}. For {ticker},
it predicts {prediction_summary}. This is {consistency_label} with the current
{macro_regime} regime and the {primary_agent} agent's view ({agent_view}).
{caveat_sentence}.
```

Placeholder sources:
- `consistency_label`: Python comparison of model direction vs. macro regime expected direction
- `caveat_sentence`: from `kb/econ/model_explanations.md` — the caveat for this model name

---

## Agent Critique Templates

### TMPL_AGENT_CRITIQUE
```
The {agent_id} agent focuses primarily on {agent_strength}. In this recommendation
for {ticker}, it {assumption_description}. From a {counterpoint_perspective} perspective,
{counterpoint}. Consider {risk_caveat} before acting on this signal.
```

---

## Fallback Instruction (no template match)
When no template key resolves fully, inject this into the LLM user message:

```
Conflict pattern: {conflict_pattern_name}
Agent outputs: {agent_outputs_json}
Provide critique in the professor's voice. Ground in ECONOMIC_FRAMEWORK.md and
ML_CRITIQUE_FRAMEWORK.md. Under 300 tokens. No buy/sell/hold signals.
```
