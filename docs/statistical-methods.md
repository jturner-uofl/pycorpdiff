# Statistical methods

This page documents *what* each metric in `pycorpdiff` is computing and
*why* the defaults are what they are. The literature is fragmented;
this is the synthesis we landed on.

## Keyness

The headline question — *which words distinguish corpus A from corpus B?*
— has been answered five ways in the corpus-linguistics literature.
`pycorpdiff` computes the four that matter and reports them side by side.

### Log-likelihood-ratio keyness (G²)

Two slightly different log-likelihood statistics circulate in the
corpus-linguistics literature for the 2-corpus / 2-state contingency
table. `pycorpdiff` exposes both via the `formula=` argument on
`compare(a, b).keyness(...)`.

**Rayson's 2-cell shortcut** (`formula="rayson"`, the default):

$$
G^2 = 2 \cdot \left(a \ln\frac{a}{E_a} + b \ln\frac{b}{E_b}\right)
$$

— summing only the *term-present* cells, where $a, b$ are the counts
in A, B and $E_a = N_A (a+b)/(N_A+N_B)$, $E_b = N_B (a+b)/(N_A+N_B)$
are the expected counts under the null of identical relative
frequencies. This is the formulation behind the UCREL Lancaster
LL Wizard (Rayson & Garside 2000) and dominates UK corpus-linguistics
practice.

**Full 4-cell Dunning G²** (`formula="dunning"`):

$$
G^2 = 2 \cdot \sum_{i=1}^{4} O_i \ln\!\left(\frac{O_i}{E_i}\right)
$$

— summing over all four cells of the 2×2 contingency table
$\{a, N_A - a; b, N_B - b\}$. This is the classical Dunning (1993)
likelihood-ratio statistic; it is the same formula NLTK's
`BigramAssocMeasures` and R's `quanteda::textstat_keyness(measure="lr")`
compute, so users who tokenise identically can cross-check across
implementations. (No live R/quanteda parity test ships with the
package — tokenisation conventions diverge enough between regex
tokenisers and `quanteda::tokens(remove_punct = TRUE)` that
term-level numerical agreement is fragile; the formula itself is
verified against the Rayson LL Wizard in the fast-tier crossval.)

Both are asymptotically $\chi^2_1$-distributed under the null. For the
near-symmetric, low-frequency cases that dominate corpus-linguistics
practice the two are typically within 1–2 % of each other; for highly
asymmetric corpora or high-frequency terms they diverge more. Use
`formula="dunning"` when cross-checking against quanteda / NLTK;
keep the default to reproduce LL-Wizard numbers.

**The sign convention.** Bare $G^2$ loses direction information.
`pycorpdiff` returns a *signed* G² — positive when the term is
overused in corpus A, negative when overused in B — and computes the
*p*-value from $|G^2|$. The unsigned form is what's $\chi^2$-distributed;
the sign is for human readability.

### Hardie's LogRatio

Effect size. With Laplace smoothing $\alpha = 0.5$:

$$
\text{LogRatio} = \log_2 \frac{(a + \alpha)/N_A}{(b + \alpha)/N_B}
$$

Smoothing keeps the ratio finite when a term is novel in one corpus.
Hardie (2014) recommends $\alpha = 0.5$ as the default; pass a different
value via `smoothing=`.

### Gabrielatos's %DIFF

The normalised percentage difference:

$$
\%\text{DIFF} = \frac{r_A - r_B}{r_B} \times 100
$$

where $r_X = (x / N_X) \cdot 10^6$ is the per-million rate. The
denominator's choice of "per million" cancels out of the ratio; we
use it for human interpretation. %DIFF is $+\infty$ when a term is
absent from B but present in A.

### BIC-approximated Bayes factor

Formula from Kass & Raftery (1995); the keyness application is
Wilson (2013):

$$
\text{BF} = \exp\!\left(\frac{|G^2| - \ln N}{2}\right)
$$

where $N = N_A + N_B$. The G² that feeds this is the same statistic
chosen via `formula=` on `keyness()` — pass `formula="dunning"` if you
want both `g2` and `bayes_factor` to use the 4-cell Dunning form.

Kass & Raftery (1995) interpretation:

- $\text{BF} > 2$  — positive evidence
- $\text{BF} > 6$  — strong evidence
- $\text{BF} > 10$ — very strong evidence
- $\text{BF} > 100$ — decisive evidence

Very large BF values overflow `float64` and surface as `inf` —
semantically correct ("evidence is essentially conclusive").

### Dispersion

A term can be "key" simply because one document overuses it.
`pycorpdiff` reports two dispersion measures alongside the keyness
score when `dispersion=True`:

- **Juilland's D** — $1 - \mathrm{CV}/\sqrt{k-1}$ where CV is the
  coefficient of variation of the per-document rates and $k$ is the
  number of documents. Range 0..1; higher = more even spread.
- **Gries's DP** — $\tfrac{1}{2} \sum_i |o_i - e_i|$ where $o_i$ is the
  document's share of the term's occurrences and $e_i$ is the document's
  share of corpus tokens. Range 0..1; lower = more even.

A keyness result with `dispersion=True` flags rows where either
corpus's $D < 0.5$ — the "one document drives it all" warning.

### Multiple-comparison correction

Vocabularies have thousands of terms; the BH-adjusted *p*-value column
(`p_adjusted`) controls the false discovery rate at 0.05 by default.
Pass `multiple_comparisons="bonferroni"` for the more conservative
family-wise correction or `"none"` to suppress adjustment.

### Defaults at a glance

| Knob                       | Default               | Reason                                                       |
|----------------------------|-----------------------|--------------------------------------------------------------|
| `method`                   | `"log_likelihood"`    | Primary, with LogRatio + BF computed alongside.              |
| `formula`                  | `"rayson"`            | Matches the UCREL LL Wizard reference. Use `"dunning"` for the classical 4-cell G² (NLTK / quanteda compatibility). |
| `min_count`                | 5                     | Below this Dunning's small-cell unreliability kicks in.      |
| `multiple_comparisons`     | `"bh"`                | FDR is the right control for exploratory term ranking.       |
| `dispersion`               | `False`               | O(V·D) memory; cheap to opt in, expensive to default on.     |
| `effect_size`              | `True`                | Always report LogRatio + %DIFF + BF alongside G².            |

## Collocations

For a target term $T$ at position $i$ in a document, the *window* is
the $w$ tokens on each side. For each (target, collocate) pair we
count:

- $f_{xy}$ — joint window co-occurrence
- $f_x$ — corpus-wide count of $T$
- $f_y$ — corpus-wide count of the collocate
- $N$ — total tokens in the corpus

Four measures from this:

| Measure   | Formula                                            | Best for                                |
|-----------|----------------------------------------------------|-----------------------------------------|
| logDice   | $14 + \log_2 \frac{2 f_{xy}}{f_x + f_y}$           | Range-bounded, corpus-size-independent. |
| PMI       | $\log_2 \frac{f_{xy} \cdot N}{f_x \cdot f_y}$      | Sensitive to rare pairs; pair with floor.|
| t-score   | $(f_{xy} - f_x f_y / N) / \sqrt{f_{xy}}$           | Favours frequent collocates.            |
| MI³       | $\log_2 \frac{f_{xy}^3 \cdot N}{f_x \cdot f_y}$    | Daille's correction to PMI's rare bias. |

The default is **logDice** (Rychlý 2008) — range-bounded above at 14,
robust to corpus size, and the de facto standard in SketchEngine.

`collocation_shift` applies Laplace smoothing ($\alpha = 0.5$) before
computing scores, so collocates absent on one side yield finite shifts
rather than $-\infty$.

## Temporal trajectories

`Tracker.over_time(freq, time_col, confidence)` reports per-period
relative frequencies with **Wilson score confidence intervals** at
`confidence=0.95` by default. Wilson is preferred to Wald for
proportion CIs because Wald collapses near $p = 0$ and $p = 1$ —
exactly where rare-term trajectories spend most of their time.

For sparse periods (zero target count), the Wilson lower bound is 0
to floating-point precision (you may see values like `5e-17`); the
upper bound shrinks with $n$. For empty periods (zero tokens) both
bounds are NaN. The score interval was originally derived by
Wilson (1927); we use the simple two-sided form. Newcombe (1998)
surveys this and related proportion-CI methods and informs the
choice over Wald for the rare-term regime.

## Changepoint detection

`TemporalTrajectory.changepoints(method, penalty)` wraps `ruptures`
(Truong, Oudre & Vayatis 2020 — the package and its accompanying
survey). Default is **PELT** (Killick, Fearnhead & Eckley 2012) with
the rbf cost — exact, penalty-controlled, and the best
general-purpose offline algorithm for trajectory-shaped data.
Alternatives: `"binseg"` (greedy, faster on long series) and
`"window"` (sliding-window scan).

Default penalty is $\ln n$ — BIC-style automatic selection. Tune
upward to reduce changepoint count.

## Bayesian online changepoint detection (BOCPD)

`TemporalTrajectory.changepoints_online(hazard=...)` implements the
Adams & MacKay (2007) online changepoint algorithm. At each new
observation $x_t$, the procedure returns a posterior over *run
length* $r_t$ — the number of periods since the most recent
changepoint:

$$
P(r_t \mid x_{1:t}) \propto \sum_{r_{t-1}} P(r_t \mid r_{t-1}) \cdot
P(x_t \mid r_{t-1}, x_{1:t-1}) \cdot P(r_{t-1} \mid x_{1:t-1})
$$

The transition $P(r_t \mid r_{t-1})$ is a constant-hazard process:
with probability $h$ (the `hazard` argument), $r_t = 0$
(changepoint); otherwise $r_t = r_{t-1} + 1$ (continuation). The
predictive likelihood $P(x_t \mid r_{t-1}, x_{1:t-1})$ uses a
Normal-Inverse-Gamma conjugate prior, giving a Student's *t*
predictive that updates analytically.

**Diagnostics.** The headline summary is the **MAP run length** —
sharp drops mark candidate changepoints. The package's
`BocpdResult.cp_probability_recent(threshold=k)` sums the leftmost
$k+1$ columns of the run-length posterior — i.e.
$P(r_t \le k \mid x_{1:t})$ — which is genuinely data-driven and
the right monitoring signal. The legacy `cp_probability` field
($P(r_t = 0 \mid x_{1:t})$) collapses to the hazard hyperparameter
under constant hazard (the changepoint prior cancels in the
normalisation) and carries no posterior information.

**Caveat on bounded data.** The Normal-Inverse-Gamma conjugate
prior is misspecified for trajectories bounded in $[0, 1]$
(relative frequencies). For proportion-trajectory applications,
either logit-transform before calling `changepoints_online`, or
prefer the offline `changepoints()` (PELT) on the rate column.

## Interrupted time series

`TemporalTrajectory.interrupted_time_series(event_date)` fits the
standard segmented-regression specification (Wagner et al. 2002):

$$
y_t = \beta_0 + \beta_1 t + \beta_2 \cdot \mathbb{1}[t \ge t_e]
     + \beta_3 (t - t_e) \cdot \mathbb{1}[t \ge t_e] + \varepsilon_t
$$

via `statsmodels.OLS`. Returned coefficients of interest:

- $\beta_2$ — **level change**: immediate step at the intervention.
- $\beta_3$ — **slope change**: how the post-period trend differs
  from the pre-period trend.

Both come with standard errors, *t*-statistics, *p*-values, and 95% CIs.

**Caveat on autocorrelation.** The default standard errors are OLS
under the homoscedasticity assumption. Rate trajectories are
typically serially autocorrelated, and Bernal, Cummins & Gasparrini
(2017) recommend Newey-West / HAC standard errors over OLS for ITS
on epidemiologic and policy time series. Treat the OLS *p*-values
as a screening signal rather than a definitive inference; bring a
HAC-corrected SE in if the conclusion hinges on a borderline
*p*-value.

## Causal impact (BSTS counterfactual)

`TemporalTrajectory.causal_impact(event_date)` fits a state-space
counterfactual model on the pre-event window and projects it
forward as the trajectory the term *would* have followed without
the event. The gap between observation and counterfactual is the
estimated effect.

We use a local-linear-trend Bayesian structural time series via
`statsmodels.UnobservedComponents`. Credible intervals on the
pointwise and cumulative effects come from Monte-Carlo simulation
against the Kalman-filter posterior.

**No-control variant.** The canonical Brodersen et al. (2015)
framework adds a regression on parallel control series with
spike-and-slab variable selection; pycorpdiff's implementation
observes only the target series itself. The univariate variant is
appropriate when no obvious control exists (the common case in
corpus-linguistic event studies) but its counterfactual relies
entirely on extrapolation of the pre-event trend — be wary of
distribution shifts unrelated to the event.

**Caveat on bounded data.** Like BOCPD, a local-linear-trend model
is unconstrained and can predict outside $[0, 1]$. For rare-term
trajectories this can drive the counterfactual mean toward zero or
negative, making `relative_effect` arbitrarily large or negative.
Logit-transform the input series or report `absolute_effect` only
when the counterfactual is near-zero.

## Forecasting (state-space ETS)

`TemporalTrajectory.forecast(horizon, logit_transform=True)`
projects the trajectory `horizon` periods forward using a
state-space exponential-smoothing model (Hyndman, Koehler, Ord &
Snyder 2008). The default `logit_transform=True` is essential for
proportion data: it maps $[0, 1]$ to $(-\infty, +\infty)$ before
fitting and back after, so forecasts and prediction intervals
respect the bounded support. With `logit_transform=False` the
unconstrained ETS may project negative rates on near-zero series —
useful only for unbounded count or rate data.

Prediction intervals come from the model's residual-based
formulation (Hyndman §6.4): 95% PIs use the empirical standard
deviation of one-step residuals, scaled by the integrated forecast
variance.

## Semantic shift

`compare(a, b).semantic_shift(target, embedder, window, align)` uses
the **averaged contextual embedding** approach (Giulianelli et al. 2020,
plus Hamilton et al. 2016 for the alignment step):

1. For every occurrence of `target` in each corpus, encode the
   surrounding window as a sentence via `embedder`.
2. Average the per-occurrence vectors into a corpus-specific centroid.
3. (Optional) Orthogonal-Procrustes-align (Schönemann 1966) the
   source centroid into the target's space — needed when the
   embedder produces independent per-corpus spaces (Hamilton-style
   word2vec), unnecessary for shared-model encoders like SBERT
   (`align="none"`, the default). Note: the current implementation
   aligns parallel rows of the per-corpus window-vector matrices
   as if they were anchor correspondences; they aren't. Until an
   anchor-vocabulary alignment path lands, `align="procrustes"`
   emits a `FutureWarning` and should be considered exploratory.
4. Report cosine distance between the centroids.

`neighborhood_drift` extends this to a top-*k* neighbour comparison.
The status partition (`shared` / `gained_in_a` / `lost_in_a`) is the
direct interpretable output.

**Caveat on averaged contextual embeddings.** Averaging contextual
embeddings is the simplest and most-cited approach to semantic
change detection, but Bommasani, Davis & Cardie (2020) demonstrate
that it can be a poor representation for change detection compared
to alternatives that preserve token-level distributional
information. Treat the cosine-distance summary as a coarse
similarity signal, not a definitive measurement of semantic shift.

## Permutation *p*-values

Optional empirical *p*-values for keyness can be requested via
`keyness(permutation_n=N)`. Documents are the unit of
exchangeability: we randomly relabel each document's corpus
membership $N$ times and recompute G² for every term, then report
the proportion of permuted G²s as extreme as the observed. The
small-$p$ floor follows Phipson & Smyth (2010): we report
$(b + 1) / (N + 1)$ rather than $b / N$, so the smallest reportable
*p* is $1 / (N + 1)$ — important when $N$ is small.

## Bootstrap confidence intervals on G²

`keyness(ci="bootstrap", n_boot=999, ci_level=0.95)` adds
`g2_ci_lower` and `g2_ci_upper` columns to the result table. The
bootstrap resamples *documents* (not tokens) with replacement,
independently on each side, preserving the original document counts.
Each resample yields a fresh per-term signed G²; the empirical
$\alpha/2$ and $1 - \alpha/2$ quantiles across `n_boot` iterations
give the percentile CI (Efron & Tibshirani 1993, §13.3). Documents
are the unit of exchangeability for the population-level inference
researchers actually want: *"if we re-collected the corpus from the
same source, what range of G² values would we see?"* A CI that
straddles zero (`g2_ci_lower < 0 < g2_ci_upper`) signals uncertainty
about the *direction* of over-use, not just the magnitude — a more
honest signal than a sub-threshold *p*-value alone.

### Per-term vs simultaneous CIs (post-selection inference)

The default percentile CI is *per-term* — calibrated for any
individual term you specify in advance, but **anti-conservative**
when read off the top of a sorted keyness table. The top-ranked
term is chosen *because* it has the largest |G²| in the observed
data; under a known null, the bootstrap CI on the top-ranked
term covers zero in only ~60-65 % of Monte-Carlo replicates,
well below the nominal 95 % (verified in
`examples/jss_case_study.ipynb` § 5.3d).

`keyness(ci="bootstrap", simultaneous_ci=True)` returns
Westfall-Young studentized-max CIs with family-wise (1 − α)
coverage across the entire vocabulary. For each bootstrap
replicate $b$ and term $t$, compute the studentized residual

$$Z_{t,b} = (G^2_{t,b} - \bar G^2_t) / s_t$$

where $\bar G^2_t$ and $s_t$ are the per-term bootstrap mean and
standard deviation. Form the per-replicate max $M_b = \max_t
|Z_{t,b}|$ and take its $1 - \alpha$ quantile, $q$. The
simultaneous CI for term $t$ is then

$$\bar G^2_t \pm q \cdot s_t.$$

Coverage is family-wise valid under arbitrary correlations
between terms (Westfall & Young 1993). Use this option when
reporting CIs on ranked keyness tables; the per-term default
is correct for fixed-term inference.

## Lexical diversity (TTR, MATTR, MTLD, HD-D)

`lexical_diversity(corpus)` reports four metrics of vocabulary
range: the naive type-token ratio plus three length-robust
alternatives. The static call returns pooled corpus-level values;
passing `freq=` (e.g. `"Y"`, `"Q"`) slices the corpus by period and
returns a per-period trajectory with optional bootstrap CIs.

The math:

- **TTR** (type-token ratio) = $|\\text{types}| / |\\text{tokens}|$.
  Length-dependent — TTR floors as text length grows because
  function words inevitably repeat. Reported for backward
  compatibility / familiarity; not defensible for cross-text
  comparison.
- **MATTR** (Moving-Average TTR, Covington & McFall 2010). Slide a
  window of $W$ tokens across the stream, compute the per-window
  TTR, average across windows. Length-robust by construction: every
  window has the same denominator so the TTR–length confound
  cancels. Default $W = 100$.
- **MTLD** (Measure of Textual Lexical Diversity, McCarthy & Jarvis
  2010). Walk the token stream, tracking the running TTR; each time
  it drops to a threshold $\\tau$ (default $0.72$, the empirically
  stable value from McCarthy 2005), record a *factor* of length
  $k$ and reset. MTLD is the mean factor length, forward-walk and
  backward-walk averaged. Conceptually: *how long can the text go
  before its diversity drops to the threshold?*
- **HD-D** (McCarthy & Jarvis 2007). For each unique type $t$ with
  count $c_t$ in a corpus of $N$ tokens, the probability of
  drawing at least one $t$ in a uniform random sample of $s$
  tokens is $1 - \\binom{N - c_t}{s} / \\binom{N}{s}$. HD-D sums
  those probabilities — the expected number of unique types in a
  size-$s$ random sample. Default $s = 42$ is the published
  convention; range is $[0, s]$. The most statistically
  principled of the three length-robust metrics.

The temporal path slices the corpus by period via the same
`time_col` + `freq` machinery used by `track().over_time(...)`,
then runs the four metrics on each period's pooled tokens. When
`ci="bootstrap"` is passed, documents within each period are
resampled with replacement and the metrics recomputed `n_boot`
times; the empirical $\\alpha/2$ and $1 - \\alpha/2$ quantiles
give the percentile CI. *Caveat:* MTLD and MATTR are
order-sensitive walks, so document-level bootstrap can mildly bias
their CIs (the point estimate occasionally falls outside the
percentile band). The CI *width* remains a useful stability
signal; TTR and HD-D, being order-independent token aggregates,
give clean percentile CIs.

## Sub-corpus balancing (Coarsened Exact Matching)

`match(a, b, on=[...])` pre-balances two corpora on document-level
covariates *before* keyness, collocation, or trajectory analysis.
Without matching, a keyness signal between two corpora that differ
systematically on a confounder (e.g. "humanising" vs "criminalising"
immigration speeches that *also* differ on year and party
distribution) reflects the joint effect of the variable of interest
*and* every confounder — exactly what causal-inference researchers
spend careers untangling.

pycorpdiff implements **Coarsened Exact Matching** (CEM; Iacus, King
& Porro 2012). Each covariate is coarsened — numeric columns into
quantile bins, categorical columns left as-is — and documents are
stratified on the joint coarsened key. Strata that contain
documents from both sides are kept; strata that don't are dropped.
Within each kept stratum the over-represented side is subsampled to
match the minority count ("k-to-k" matching), so the resulting
matched slices have equal stratum-level counts on both sides.

The standard CEM diagnostic — *L1 imbalance* on each covariate, half
the sum of absolute differences between the two sides' empirical
marginal distributions — is reported on the returned ``MatchResult``
before and after matching. ``l1_post`` ≪ ``l1_pre`` is what matching
buys you. Match results are fully reproducible under a fixed
``seed``.

CEM is the natural fit for corpus linguistics: no propensity model
is needed (the model would be opaque to corpus linguists), it
handles the mixed-categorical metadata corpus archives actually
have (party, year-bucket, topic, speaker-role) without contortion,
and the matched slices plug straight into every other analytical
verb — `pcd.compare(m.a_matched, m.b_matched).keyness()` is the
canonical use, but `compare.collocation_shift`, `track`, and
`against_baseline` all work the same way.

## Burstiness detection (Kleinberg 1999)

`TemporalTrajectory.burstiness()` segments a target's per-period
rate into burst-intensity states. Where `changepoints()` answers
*"when did the rate change?"* (segmentation by location),
burstiness answers *"when was the rate elevated and by how
much?"* (per-period intensity labelling).

The algorithm fits a multi-state automaton where state $i$ has rate
$p_0 \cdot s^i$ — $p_0$ is the overall base rate of the term across
all periods, and $s > 1$ is the **burst factor** (default $s = 2$,
following Kleinberg's recommendation). Per-period observation cost
is the negative log-Binomial likelihood under state $i$'s rate;
transition cost is $(j - i) \cdot \gamma \cdot \log T$ when
escalating to a higher state and zero when de-escalating
("burst-up is costly, burst-down is free"). A standard Viterbi pass
over the per-period × per-state cost matrix produces the
minimum-cost state sequence. The state count is capped at
`n_states` (default 5) — corpus data rarely populates beyond state
2 or 3.

The returned `BurstinessResult` carries (a) per-period state
labels and (b) a per-burst summary table (one row per maximal
contiguous run of state ≥ 1). The `.plot()` method overlays the
state intensity onto the trajectory line.

## Reference-corpus keyness (`against_baseline`)

`against_baseline(corpus, "gutenberg_fiction")` compares the user's
corpus against a pre-computed reference frequency baseline, returning
a `KeynessResult` with the same shape as `compare(a, b).keyness()`.
The math is identical (signed G² with the same `formula=` toggle); the
operational difference is that the reference side is an aggregated
`(term, count)` frequency list with a corpus total, not a full
:class:`Corpus`. This is the canonical setup in lexicography and
discourse analysis (BNC, COCA, in-house reference corpora): aggregated
frequency lists are typically two to three orders of magnitude smaller
than the source text and side-step reference-corpus licence
complications.

Bundled baselines are listed by `pycorpdiff.list_baselines()`. The
0.1.0a14 release ships one starter:

- **`"gutenberg_fiction"`** — five Project Gutenberg English-fiction
  texts (Austen, Carroll, Doyle, Shelley, Stoker; 1813–1897), ~500K
  tokens, ~11K types after a hapax-legomena threshold of 2. Public
  Domain. Useful as an out-of-the-box "general English fiction"
  reference; clearly inappropriate where the 19th-c. contrast is
  itself the signal.

User-supplied baselines come from
`pycorpdiff.baseline_from_corpus(corpus)` — aggregate any
`pycorpdiff.Corpus` (a BNC slice, a HuggingFace dataset, an in-house
crawl) once, then reuse it. Because the reference side has no
per-document structure on hand, dispersion, permutation, and
bootstrap CIs are unavailable through `against_baseline`; run a full
`compare(a, b)` against a real `Corpus` if you need those.

## References

- Adams, R. P., & MacKay, D. J. C. (2007). Bayesian online changepoint detection. arXiv:0710.3742.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. *J. Royal Statistical Society B*, 57(1), 289–300.
- Brodersen, K. H., et al. (2015). Inferring causal impact using Bayesian structural time-series models. *Annals of Applied Statistics*, 9(1), 247–274.
- Church, K. W., & Hanks, P. (1990). Word association norms, mutual information, and lexicography. *Computational Linguistics*, 16(1), 22–29.
- Church, K., et al. (1991). In *Lexical Acquisition*, 115–164.
- Covington, M. A., & McFall, J. D. (2010). Cutting the Gordian knot: the moving-average type-token ratio (MATTR). *Journal of Quantitative Linguistics*, 17(2), 94–100.
- Daille, B. (1994). PhD thesis, Université Paris 7.
- Dunning, T. (1993). *Computational Linguistics*, 19(1), 61–74.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Gabrielatos, C. (2018). In *Corpus Approaches to Discourse*, 225–258.
- Giulianelli, M., Del Tredici, M., & Fernández, R. (2020). Analysing lexical semantic change with contextualised word representations. In *Proceedings of ACL 2020*, 3960–3973.
- Gries, S. Th. (2008). *IJCL*, 13(4), 403–437.
- Hamilton, W. L., et al. (2016). In *Proceedings of ACL 2016*.
- Bernal, J. L., Cummins, S., & Gasparrini, A. (2017). Interrupted time series regression for the evaluation of public health interventions: a tutorial. *International Journal of Epidemiology*, 46(1), 348–355.
- Bommasani, R., Davis, K., & Cardie, C. (2020). Interpreting Pretrained Contextualized Representations via Reductions to Static Embeddings. In *Proceedings of ACL 2020*, 4758–4781.
- Hardie, A. (2014). Log Ratio — an informal introduction. CASS blog post, https://cass.lancs.ac.uk/log-ratio-an-informal-introduction/
- Hyndman, R. J., Koehler, A. B., Ord, J. K., & Snyder, R. D. (2008). *Forecasting with Exponential Smoothing: The State Space Approach*. Springer.
- Iacus, S. M., King, G., & Porro, G. (2012). Causal inference without balance checking: Coarsened exact matching. *Political Analysis*, 20(1), 1–24.
- Juilland, A., & Chang-Rodríguez, E. (1964). *Frequency Dictionary of Spanish Words*.
- Kass, R. E., & Raftery, A. E. (1995). *JASA*, 90(430), 773–795.
- Killick, R., Fearnhead, P., & Eckley, I. A. (2012). *JASA*, 107(500), 1590–1598.
- Kleinberg, J. (2003). Bursty and hierarchical structure in streams. *Data Mining and Knowledge Discovery*, 7(4), 373–397.
- McCarthy, P. M. (2005). *An assessment of the range and usefulness of lexical diversity measures and the potential of the measure of textual, lexical diversity (MTLD)*. PhD dissertation, University of Memphis.
- McCarthy, P. M., & Jarvis, S. (2007). vocd: A theoretical and empirical evaluation. *Language Testing*, 24(4), 459–488.
- McCarthy, P. M., & Jarvis, S. (2010). MTLD, vocd-D, and HD-D: A validation study of sophisticated approaches to lexical diversity assessment. *Behavior Research Methods*, 42(2), 381–392.
- Newcombe, R. G. (1998). Two-sided confidence intervals for the single proportion: comparison of seven methods. *Statistics in Medicine*, 17(8), 857–872.
- Phipson, B., & Smyth, G. K. (2010). Permutation *P*-values should never be zero. *Statistical Applications in Genetics and Molecular Biology*, 9(1), Article 39.
- Rayson, P., & Garside, R. (2000). Comparing corpora using frequency profiling. In *Proceedings of the Workshop on Comparing Corpora* (ACL 2000), 1–6.
- Rychlý, P. (2008). In *Proceedings of RASLAN 2008*.
- Schönemann, P. H. (1966). A generalized solution of the orthogonal Procrustes problem. *Psychometrika*, 31(1), 1–10.
- Truong, C., Oudre, L., & Vayatis, N. (2020). Selective review of offline change point detection methods. *Signal Processing*, 167, 107299.
- Wagner, A. K., et al. (2002). *J. Clin. Pharm. Ther.*, 27(4), 299–309.
- Wilson, A. (2013). Embracing Bayes factors for key item analysis in corpus linguistics. In *New Approaches to the Study of Linguistic Variability*, 3–11.
- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *JASA*, 22(158), 209–212.
