# Statistical methods

This page documents *what* each metric in `pycorpdiff` is computing and
*why* the defaults are what they are. The literature is fragmented;
this is the synthesis we landed on.

## Keyness

The headline question — *which words distinguish corpus A from corpus B?*
— has been answered five ways in the corpus-linguistics literature.
`pycorpdiff` computes the four that matter and reports them side by side.

### Dunning's log-likelihood (G²)

The default. For a term with counts $a$ in corpus A (of size $N_A$) and
$b$ in corpus B (of size $N_B$):

$$
G^2 = 2 \cdot \sum_i O_i \ln\!\left(\frac{O_i}{E_i}\right)
$$

where $O_i$ are the four observed cell counts of the 2×2 contingency
table $\{a, N_A - a; b, N_B - b\}$ and $E_i$ are their expected counts
under the null of identical relative frequencies. Under the null, $G^2$
is approximately $\chi^2$-distributed with 1 degree of freedom.

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

### Wilson's Bayes factor

The BIC-approximated Bayes factor:

$$
\text{BF} = \exp\!\left(\frac{|G^2| - \ln N}{2}\right)
$$

where $N = N_A + N_B$. Kass & Raftery (1995) interpretation: BF > 10 is
strong evidence, BF > 100 is decisive. Very strong evidence overflows
to `inf` — semantically correct ("essentially conclusive").

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
exactly; the upper bound shrinks with $n$. For empty periods (zero
tokens) both bounds are NaN.

## Changepoint detection

`TemporalTrajectory.changepoints(method, penalty)` wraps `ruptures`.
Default is **PELT** (Killick et al. 2012) with the rbf cost — exact,
penalty-controlled, and the best general-purpose offline algorithm for
trajectory-shaped data. Alternatives: `"binseg"` (greedy, faster on long
series) and `"window"` (sliding-window scan).

Default penalty is $\ln n$ — BIC-style automatic selection. Tune
upward to reduce changepoint count.

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

## Semantic shift

`compare(a, b).semantic_shift(target, embedder, window, align)` uses
the **averaged contextual embedding** approach (Giulianelli et al. 2020,
plus Hamilton et al. 2016 for the alignment step):

1. For every occurrence of `target` in each corpus, encode the
   surrounding window as a sentence via `embedder`.
2. Average the per-occurrence vectors into a corpus-specific centroid.
3. (Optional) Orthogonal-Procrustes-align the source centroid into the
   target's space — needed when the embedder produces independent
   per-corpus spaces (Hamilton-style word2vec), unnecessary for shared-
   model encoders like SBERT (`align="none"`, the default).
4. Report cosine distance between the centroids.

`neighborhood_drift` extends this to a top-*k* neighbour comparison.
The status partition (`shared` / `gained_in_a` / `lost_in_a`) is the
direct interpretable output.

## References

- Dunning, T. (1993). *Computational Linguistics*, 19(1), 61–74.
- Hardie, A. (2014). *Log Ratio*. CASS technical note.
- Gabrielatos, C. (2018). In *Corpus Approaches to Discourse*, 225–258.
- Wilson, A. (2013). In *New Approaches to the Study of Linguistic Variability*, 3–11.
- Kass, R. E., & Raftery, A. E. (1995). *JASA*, 90(430), 773–795.
- Juilland, A., & Chang-Rodríguez, E. (1964). *Frequency Dictionary of Spanish Words*.
- Gries, S. Th. (2008). *IJCL*, 13(4), 403–437.
- Rychlý, P. (2008). In *Proceedings of RASLAN 2008*.
- Church, K., et al. (1991). In *Lexical Acquisition*, 115–164.
- Daille, B. (1994). PhD thesis, Université Paris 7.
- Wagner, A. K., et al. (2002). *J. Clin. Pharm. Ther.*, 27(4), 299–309.
- Killick, R., Fearnhead, P., & Eckley, I. A. (2012). *JASA*, 107(500), 1590–1598.
- Hamilton, W. L., et al. (2016). In *Proceedings of ACL 2016*.
- Giulianelli, M., et al. (2020). In *Proceedings of ACL 2020*.
- Newcombe, R. G. (1998). *Statistics in Medicine*, 17(8), 857–872.
- Schönemann, P. H. (1966). *Psychometrika*, 31(1), 1–10.
