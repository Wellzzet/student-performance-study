"""What Drives Students' Final Math Grades? — interactive statistical study (Streamlit).
Run locally:  streamlit run app.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from scipy.optimize import minimize

st.set_page_config(page_title="Student Performance Study", page_icon="📊", layout="wide")

# ---------------------------------------------------------------- design system: white · black · green (+ blue)
INK, MUTED, LINE, BG_SOFT = "#0B0B0C", "#6B7280", "#E5E7EB", "#F6F7F6"
GREEN, BLUE = "#15803D", "#2563EB"          # primary / secondary data colours (colour-blind-safe pair)
GREEN_SOFT = "#DCFCE7"
ORANGE, RED = BLUE, INK                     # legacy aliases
CMAP = LinearSegmentedColormap.from_list("bgg", [BLUE, "#F3F4F6", GREEN])   # diverging: blue ← gray → green

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.edgecolor": "#D1D5DB", "axes.labelcolor": MUTED, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 10,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#EEF0F2", "grid.linewidth": .8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "patch.linewidth": 0, "lines.linewidth": 2,
})

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stMarkdown, p, li, label, input, button, textarea {{ font-family: 'Inter', sans-serif; }}
.block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
h1, h2, h3 {{ font-family: 'Inter', sans-serif !important; color: {INK}; letter-spacing: -0.02em; }}
h2, h3 {{ font-weight: 700 !important; }}
[data-testid="stSidebar"] {{ background: {INK}; }}
[data-testid="stSidebar"] * {{ color: #F3F4F6 !important; }}
[data-testid="stSidebar"] .stRadio label {{ padding: 7px 12px; border-radius: 9px; width: 100%; margin-bottom: 2px; }}
[data-testid="stSidebar"] .stRadio label:hover {{ background: #1F2937; }}
[data-testid="stSidebar"] .stRadio label:has(input:checked) {{ background: {GREEN}; }}
[data-testid="stSidebar"] .stRadio label > div:first-child {{ display: none; }}
.side-brand {{ font-size: 1.2rem; font-weight: 800; letter-spacing: -0.02em; margin: 4px 0 4px; }}
.side-brand span {{ color: #4ADE80 !important; }}
.side-note {{ font-size: .78rem; opacity: .65; line-height: 1.5; }}
.hero {{ background: {INK}; color: white; border-radius: 20px; padding: 34px 38px; margin-bottom: 26px;
        background-image: radial-gradient(circle at 95% 0%, rgba(21,128,61,.6), transparent 45%); }}
.hero .kicker {{ display: inline-block; background: {GREEN}; color: white; font-size: .72rem; font-weight: 700;
        letter-spacing: .08em; text-transform: uppercase; padding: 5px 12px; border-radius: 999px; margin-bottom: 14px; }}
.hero h1 {{ color: white !important; font-size: 2.3rem; font-weight: 800; line-height: 1.12; margin: 0 0 10px 0; padding: 0; }}
.hero p {{ color: #D1D5DB; font-size: 1.02rem; margin: 0; max-width: 780px; line-height: 1.55; }}
[data-testid="stMetric"] {{ background: white; border: 1px solid {LINE}; border-radius: 14px; padding: 16px 18px;
        box-shadow: 0 1px 2px rgba(0,0,0,.04); }}
[data-testid="stMetricLabel"] p {{ color: {MUTED} !important; font-size: .82rem; font-weight: 500; }}
[data-testid="stMetricValue"] {{ color: {INK}; font-weight: 800; letter-spacing: -0.02em; }}
.card {{ border: 1px solid {LINE}; border-radius: 14px; padding: 18px 20px; background: white; height: 100%;
        box-shadow: 0 1px 2px rgba(0,0,0,.04); }}
.card .num {{ color: {GREEN}; font-weight: 800; font-size: .78rem; letter-spacing: .08em; }}
.card h4 {{ margin: 4px 0 6px 0; font-size: 1rem; color: {INK}; font-weight: 700; }}
.card p {{ margin: 0; color: {MUTED}; font-size: .9rem; line-height: 1.45; }}
.finding {{ border-left: 4px solid {GREEN}; background: {BG_SOFT}; border-radius: 0 12px 12px 0; padding: 14px 18px; margin: 10px 0 18px; }}
.pill {{ display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: .74rem; font-weight: 700; }}
.pill.yes {{ background: {GREEN_SOFT}; color: {GREEN}; }}
.pill.no {{ background: #F3F4F6; color: {MUTED}; }}
[data-testid="stDataFrame"], [data-testid="stTable"] {{ border-radius: 12px; overflow: hidden; }}
.stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


def hero(title, kicker, text=""):
    st.markdown(f'<div class="hero"><span class="kicker">{kicker}</span><h1>{title}</h1><p>{text}</p></div>',
                unsafe_allow_html=True)


def finding(html):
    st.markdown(f'<div class="finding">{html}</div>', unsafe_allow_html=True)


def show(f):
    f.tight_layout()
    st.pyplot(f, width="stretch")
    plt.close(f)


@st.cache_data
def load():
    base = Path(__file__).parent
    path = base / "data" / "student-mat.csv"
    if not path.exists():
        path = base / "student-mat.csv"
    d = pd.read_csv(path, sep=";")
    for c in ["G1", "G2", "G3"]:
        d[c] = pd.to_numeric(d[c])
    d["passed"] = d["G3"] >= 10
    return d


df = load()
N = len(df)


def t_ci(x, conf=0.95):
    x = np.asarray(x, float); n = len(x); se = x.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf((1 + conf) / 2, n - 1)
    return x.mean() - t * se, x.mean() + t * se


def cohen_d(x1, x2):
    n1, n2 = len(x1), len(x2)
    sp = np.sqrt(((n1 - 1) * x1.var(ddof=1) + (n2 - 1) * x2.var(ddof=1)) / (n1 + n2 - 2))
    return (x1.mean() - x2.mean()) / sp


def fig(w=9, h=4, ncols=1):
    return plt.subplots(1, ncols, figsize=(w, h))


# ---------------------------------------------------------------- sidebar
st.sidebar.markdown('<div class="side-brand">Student <span>Performance</span></div>'
                    '<div class="side-note">Probability Theory &amp; Mathematical Statistics<br>Narxoz University</div>',
                    unsafe_allow_html=True)
st.sidebar.markdown("&nbsp;")
page = st.sidebar.radio("Section", [
    "1. Overview", "2. Data & descriptive stats", "3. Probability & Bayes", "4. Estimation (MoM & MLE)",
    "5. Confidence intervals", "6. Hypothesis tests", "7. Correlation & regression", "8. CLT simulator", "9. Conclusions"],
    label_visibility="collapsed")
st.sidebar.markdown(f'<div class="side-note" style="margin-top:18px">n = {N} students · α = 0.05<br>'
                    'Data: Cortez &amp; Silva (2008), UCI <i>Student Performance</i></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- 1
if page.startswith("1"):
    hero("What drives students’ final math grades?", "Research study · n = 395", "A statistical study of 395 secondary-school students using the tools of the course: probability, Bayes’ theorem, MoM &amp; MLE, confidence intervals, hypothesis tests and the Central Limit Theorem.")
    c = st.columns(4)
    c[0].metric("Students", N)
    c[1].metric("Mean final grade", f"{df.G3.mean():.2f} / 20")
    c[2].metric("Pass rate (G3 ≥ 10)", f"{df.passed.mean():.1%}")
    c[3].metric("Variables", df.shape[1] - 1)
    st.markdown("### Research questions")
    rqs = [("How likely is a student to pass?", "Conditional probability, Bayes’ theorem"),
           ("Which distributions fit the data?", "MoM, MLE, χ² goodness-of-fit, AIC"),
           ("What are the true mean and pass rate?", "t, χ², Wald and Wilson intervals"),
           ("Which factors matter?", "Welch t, ANOVA, Mann–Whitney, χ² independence"),
           ("Do early grades predict the final?", "Correlation, linear regression"),
           ("Does the CLT hold for skewed data?", "Simulation, bootstrap")]
    for row in range(2):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            q, m = rqs[row * 3 + j]
            col.markdown(f'<div class="card"><div class="num">RQ{row * 3 + j + 1}</div><h4>{q}</h4><p>{m}</p></div>',
                         unsafe_allow_html=True)
        st.markdown("")
    finding("<b>Key findings.</b> Past failures and higher-education plans have the strongest association with the "
            "final grade; study time only a weak one; absences follow a negative binomial — not a Poisson — distribution.")

# ---------------------------------------------------------------- 2
elif page.startswith("2"):
    hero("Data &amp; descriptive statistics", "02 · Data", "Two Portuguese secondary schools, 2005–06. G3 is the final grade (0–20); a student passes if G3 ≥ 10.")
    with st.expander("Show raw data"):
        st.dataframe(df, width="stretch")
    num = ["G1", "G2", "G3", "absences", "age", "studytime", "failures"]
    desc = df[num].agg(["mean", "median", "var", "std", "min", "max"]).T
    desc["skewness"] = df[num].apply(stats.skew)
    st.dataframe(desc.round(2), width="stretch")
    var = st.selectbox("Variable to plot", ["G3", "G1", "G2", "absences", "age", "studytime", "failures", "goout", "Walc"])
    f, ax = fig(10, 3.8)
    vals = df[var]
    ax.hist(vals, bins=np.arange(vals.min() - .5, vals.max() + 1.5, 1), color=GREEN, edgecolor="white")
    ax.axvline(vals.mean(), color=RED, ls="--", label=f"mean = {vals.mean():.2f}")
    ax.axvline(vals.median(), color=BLUE, ls=":", label=f"median = {vals.median():.1f}")
    ax.set(title=var, ylabel="students"); ax.legend(); show(f)
    st.caption("38 students have G3 = 0 while all have G1 > 0 — most likely they dropped out / missed the exam.")

# ---------------------------------------------------------------- 3
elif page.startswith("3"):
    hero("Probability &amp; Bayes’ theorem", "03 · Probability", "Empirical and conditional probabilities, the law of total probability and Bayes’ theorem — computed live from the sample.")
    st.latex(r"P(A\mid B)=\frac{P(A\cap B)}{P(B)},\qquad P(B\mid A)=\frac{P(A\mid B)P(B)}{P(A)}")
    factor = st.selectbox("Condition B", ["failures ≥ 1", "higher = no", "internet = no", "romantic = yes",
                                          "studytime ≥ 3", "goout ≥ 4", "absences > 10", "sex = M"])
    masks = {"failures ≥ 1": df.failures >= 1, "higher = no": df.higher == "no", "internet = no": df.internet == "no",
             "romantic = yes": df.romantic == "yes", "studytime ≥ 3": df.studytime >= 3, "goout ≥ 4": df.goout >= 4,
             "absences > 10": df.absences > 10, "sex = M": df.sex == "M"}
    B = masks[factor]; A = df.passed
    pA, pB = A.mean(), B.mean(); pAB = A[B].mean(); pAnB = A[~B].mean()
    pB_notA = (1 - pAB) * pB / (1 - pA)
    c = st.columns(4)
    c[0].metric("P(pass)", f"{pA:.3f}")
    c[1].metric("P(B)", f"{pB:.3f}")
    c[2].metric("P(pass | B)", f"{pAB:.3f}", f"{pAB - pA:+.3f}")
    c[3].metric("P(pass | not B)", f"{pAnB:.3f}")
    st.markdown(f"**Law of total probability:** {pAB:.3f}·{pB:.3f} + {pAnB:.3f}·{1 - pB:.3f} = **{pAB * pB + pAnB * (1 - pB):.3f}** = P(pass) ✓")
    st.markdown(f"**Bayes:** P(B | not pass) = (1−{pAB:.3f})·{pB:.3f} / {1 - pA:.3f} = **{pB_notA:.3f}** "
                f"(vs P(B) = {pB:.3f}).")
    indep = abs((A & B).mean() - pA * pB) < 0.005
    st.markdown(f"P(pass ∩ B) = {(A & B).mean():.3f}, P(pass)·P(B) = {pA * pB:.3f} → "
                + ("approximately **independent** in the sample." if indep else "**not independent** in the sample."))
    ct = pd.crosstab(B.map({True: factor, False: "not " + factor}), A.map({True: "pass", False: "fail"}), margins=True)
    st.dataframe(ct)

# ---------------------------------------------------------------- 4
elif page.startswith("4"):
    hero("Method of moments vs maximum likelihood", "04 · Estimation", "Which distribution fits grades and absences — and how MoM and MLE estimates compare.")
    tab1, tab2 = st.tabs(["Final grades — Normal", "Absences — Poisson vs Negative binomial"])
    with tab1:
        g = df.loc[df.G3 > 0, "G3"].values.astype(float)
        st.latex(r"\ell(\mu,\sigma)=-\tfrac n2\ln(2\pi\sigma^2)-\tfrac1{2\sigma^2}\sum(x_i-\mu)^2\ \Rightarrow\ "
                 r"\hat\mu=\bar x,\ \hat\sigma^2=\tfrac1n\sum(x_i-\bar x)^2")
        res = minimize(lambda t: -stats.norm.logpdf(g, t[0], np.exp(t[1])).sum(), [5, 0])
        mu, sd = res.x[0], np.exp(res.x[1])
        c = st.columns(3)
        c[0].metric("MoM μ̂ / σ̂", f"{g.mean():.3f} / {g.std():.3f}")
        c[1].metric("MLE μ̂ / σ̂ (numerical)", f"{mu:.3f} / {sd:.3f}")
        c[2].metric("n (G3 > 0)", len(g))
        f, ax = fig(10, 3.8); x = np.linspace(0, 21, 300)
        ax.hist(g, bins=np.arange(.5, 21.5, 1), density=True, color=GREEN, edgecolor="white")
        ax.plot(x, stats.norm.pdf(x, mu, sd), color=RED, lw=2, label=f"N({mu:.2f}, {sd:.2f}²)"); ax.legend(); show(f)
        sw = stats.shapiro(g)
        st.markdown(f"Shapiro–Wilk p = {sw.pvalue:.1e}: the normal model is a good **approximation** but not exact — "
                    "grades cluster just above the pass mark (10–11) and are integers.")
    with tab2:
        a = df.absences.values; m, s2 = a.mean(), a.var(ddof=1)
        st.markdown(f"Mean = **{m:.2f}**, variance = **{s2:.2f}** → variance/mean = **{s2 / m:.1f}**. "
                    "Poisson requires mean = variance, so the data are strongly **overdispersed**.")
        res = minimize(lambda t: -stats.nbinom.logpmf(a, np.exp(t[0]), 1 / (1 + np.exp(-t[1]))).sum(), [0, 0])
        r, p = np.exp(res.x[0]), 1 / (1 + np.exp(-res.x[1]))
        llp = stats.poisson.logpmf(a, m).sum()
        st.dataframe(pd.DataFrame({
            "model": ["Poisson (MLE)", "NegBin (MoM)", "NegBin (MLE)"],
            "parameters": [f"λ = {m:.3f}", f"r = {m**2 / (s2 - m):.3f}, p = {m / s2:.3f}", f"r = {r:.3f}, p = {p:.3f}"],
            "log-likelihood": [llp, stats.nbinom.logpmf(a, m**2 / (s2 - m), m / s2).sum(), -res.fun],
            "AIC": [2 - 2 * llp, 4 - 2 * stats.nbinom.logpmf(a, m**2 / (s2 - m), m / s2).sum(), 4 + 2 * res.fun]}).round(2),
            hide_index=True)
        k = np.arange(0, 31); f, ax = fig(10, 3.8)
        ax.bar(k, [np.mean(a == i) for i in k], color="#D1D5DB", label="data")
        ax.plot(k, stats.poisson.pmf(k, m), "o-", color=BLUE, ms=5, label="Poisson")
        ax.plot(k, stats.nbinom.pmf(k, r, p), "s-", color=GREEN, ms=5, label="Negative binomial"); ax.legend(); show(f)
        finding("<b>Negative binomial wins:</b> its AIC is about 2 000 lower than the Poisson model.")

# ---------------------------------------------------------------- 5
elif page.startswith("5"):
    hero("Confidence intervals", "05 · Inference", "Interval estimates for the mean, the variance and the pass rate. Move the slider to change the confidence level.")
    conf = st.slider("Confidence level", .80, .99, .95, .01)
    z = stats.norm.ppf((1 + conf) / 2)
    var = st.selectbox("Variable for the mean", ["G3", "G1", "G2", "absences", "studytime"])
    x = df[var].astype(float); lo, hi = t_ci(x, conf)
    st.latex(r"\bar x \pm t_{\alpha/2,n-1}\frac{s}{\sqrt n}")
    st.markdown(f"**{conf:.0%} CI for mean {var}:** {x.mean():.3f} ∈ [**{lo:.3f}**, **{hi:.3f}**]")
    n = N; s2 = df.G3.var(ddof=1); al = 1 - conf
    st.markdown(f"**{conf:.0%} CI for variance of G3:** [{(n - 1) * s2 / stats.chi2.ppf(1 - al / 2, n - 1):.2f}, "
                f"{(n - 1) * s2 / stats.chi2.ppf(al / 2, n - 1):.2f}] (point estimate {s2:.2f})")
    ph = df.passed.mean(); se = np.sqrt(ph * (1 - ph) / n)
    c_ = (ph + z**2 / (2 * n)) / (1 + z**2 / n); h = z * np.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    st.markdown(f"**{conf:.0%} CI for pass rate:** Wald [{ph - z * se:.3f}, {ph + z * se:.3f}], Wilson [{c_ - h:.3f}, {c_ + h:.3f}]")

    st.subheader("What does the confidence level mean?")
    m = st.slider("Sample size per interval", 10, 200, 40, 10)
    seed = st.number_input("Random seed", value=2026, step=1)
    rng = np.random.default_rng(int(seed)); pop = df.G3.values; mu = pop.mean(); cov = 0
    f, ax = fig(11, 4)
    for i in range(100):
        l, u = t_ci(rng.choice(pop, m), conf); ok = l <= mu <= u; cov += ok
        ax.plot([i, i], [l, u], color=GREEN if ok else INK, lw=1.6 if ok else 2.6, solid_capstyle="round")
    ax.axhline(mu, color="black", ls="--"); ax.set_title(f"{cov} of 100 intervals contain the true mean {mu:.2f}")
    show(f)

# ---------------------------------------------------------------- 6
elif page.startswith("6"):
    hero("Hypothesis tests", "06 · Testing", "H₀: the factor is not associated with the final grade · H₁: it is (two-sided) · α = 0.05")
    binary = {"higher": ("yes", "no"), "internet": ("yes", "no"), "sex": ("M", "F"), "romantic": ("no", "yes"),
              "address": ("U", "R"), "paid": ("yes", "no"), "schoolsup": ("no", "yes"), "famsup": ("yes", "no"),
              "activities": ("yes", "no")}
    multi = ["failures", "studytime", "goout", "Walc", "Medu", "traveltime", "health"]
    col = st.selectbox("Factor", list(binary) + multi)
    if col in binary:
        a1, a2 = binary[col]; x1, x2 = df.loc[df[col] == a1, "G3"], df.loc[df[col] == a2, "G3"]
        t = stats.ttest_ind(x1, x2, equal_var=False); u = stats.mannwhitneyu(x1, x2)
        chi = stats.chi2_contingency(pd.crosstab(df[col], df.passed))
        c = st.columns(4)
        c[0].metric(f"mean G3 ({a1}), n={len(x1)}", f"{x1.mean():.2f}")
        c[1].metric(f"mean G3 ({a2}), n={len(x2)}", f"{x2.mean():.2f}")
        c[2].metric("Cohen's d", f"{cohen_d(x1, x2):.2f}")
        c[3].metric("Welch t", f"{t.statistic:.2f}")
        tbl = pd.DataFrame({"test": ["Welch t-test", "Mann–Whitney U", "χ² independence (vs pass/fail)"],
                            "p-value": [t.pvalue, u.pvalue, chi[1]]})
    else:
        groups = [v.G3.values for _, v in df.groupby(col)]
        f_ = stats.f_oneway(*groups); kw = stats.kruskal(*groups); chi = stats.chi2_contingency(pd.crosstab(df[col], df.passed))
        st.markdown(f"ANOVA F = **{f_.statistic:.2f}**")
        tbl = pd.DataFrame({"test": ["One-way ANOVA", "Kruskal–Wallis", "χ² independence (vs pass/fail)"],
                            "p-value": [f_.pvalue, kw.pvalue, chi[1]]})
    tbl["decision"] = np.where(tbl["p-value"] < .05, "Reject H₀", "Fail to reject H₀")
    tbl["Bonferroni (α/10)"] = np.where(tbl["p-value"] < .005, "Reject H₀", "Fail to reject H₀")
    st.dataframe(tbl.style.format({"p-value": "{:.2e}"}), hide_index=True, width="stretch")
    grp = df.groupby(col).G3
    means = grp.mean(); err = grp.apply(lambda x: x.mean() - t_ci(x)[0] if len(x) > 1 else 0)
    f, axs = fig(12, 3.8, 2)
    axs[0].bar(means.index.astype(str), means, yerr=err, capsize=5, color=GREEN, ecolor=INK); axs[0].set_title("Mean G3 ± 95 % CI")
    pr = df.groupby(col).passed.mean(); axs[1].bar(pr.index.astype(str), pr, color=BLUE)
    axs[1].set_title("Pass rate"); axs[1].set_ylim(0, 1); show(f)
    st.caption("We run many tests, so some 'significant' results may be false positives. The Bonferroni column uses α = 0.005.")

# ---------------------------------------------------------------- 7
elif page.startswith("7"):
    hero("Correlation &amp; regression", "07 · Relationships", "How strongly do early grades and habits move together with the final grade?")
    cols = ["G1", "G2", "G3", "failures", "studytime", "absences", "goout", "Walc", "age", "Medu"]
    method = st.radio("Correlation", ["spearman", "pearson"], horizontal=True)
    corr = df[cols].corr(method=method)
    f, ax = fig(7, 6); im = ax.imshow(corr, cmap=CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45); ax.set_yticks(range(len(cols)), cols); ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im); show(f)
    xv = st.selectbox("Predictor of G3", ["G1", "G2", "failures", "absences", "studytime", "goout", "age"])
    lr = stats.linregress(df[xv], df.G3)
    st.latex(rf"\widehat{{G3}} = {lr.intercept:.3f} {'+' if lr.slope >= 0 else '-'} {abs(lr.slope):.3f}\cdot {xv}"
             rf"\qquad r={lr.rvalue:.3f},\ R^2={lr.rvalue**2:.3f},\ p={lr.pvalue:.1e}")
    rng = np.random.default_rng(1); j = rng.uniform(-.25, .25, N)
    f, ax = fig(9, 4); ax.scatter(df[xv] + j, df.G3 + j, s=14, alpha=.45, color=BLUE, edgecolors="none")
    xx = np.linspace(df[xv].min(), df[xv].max(), 10); ax.plot(xx, lr.intercept + lr.slope * xx, color=RED, lw=2)
    ax.set(xlabel=xv, ylabel="G3"); show(f)

# ---------------------------------------------------------------- 8
elif page.startswith("8"):
    hero("Central Limit Theorem", "08 · Simulation", "Draw thousands of samples from a skewed variable and watch the sample means become normal.")
    st.latex(r"\bar X_n \;\approx\; N\!\left(\mu,\ \frac{\sigma^2}{n}\right)\quad\text{for large } n")
    var = st.selectbox("Population (our data)", ["absences", "G3", "failures", "Walc"])
    n = st.slider("Sample size n", 1, 200, 30)
    reps = st.select_slider("Number of samples", [1000, 5000, 10000, 20000], 10000)
    pop = df[var].values.astype(float); rng = np.random.default_rng(0)
    means = rng.choice(pop, (reps, n)).mean(axis=1)
    f, axs = fig(13, 4, 2)
    axs[0].hist(pop, bins=30, color="#9CA3AF", edgecolor="white"); axs[0].set_title(f"Population: skew = {stats.skew(pop):.2f}")
    axs[1].hist(means, bins=50, density=True, color=GREEN, edgecolor="white")
    xs = np.linspace(means.min(), means.max(), 200)
    axs[1].plot(xs, stats.norm.pdf(xs, pop.mean(), pop.std() / np.sqrt(n)), color=RED, lw=2, label="CLT normal")
    axs[1].set_title(f"Sample means, n = {n}: skew = {stats.skew(means):.2f}"); axs[1].legend(); show(f)
    c = st.columns(3)
    c[0].metric("SD of sample means", f"{means.std():.3f}")
    c[1].metric("Theory σ/√n", f"{pop.std() / np.sqrt(n):.3f}")
    c[2].metric("Mean of sample means", f"{means.mean():.3f}", f"μ = {pop.mean():.3f}", delta_color="off")

# ---------------------------------------------------------------- 9
else:
    hero("Conclusions", "09 · Results", "What the data say — and what they cannot say.")
    items = [("Past failures", "P(pass | ≥1 failure) = 0.37 vs 0.75 without; mean grade ≈ 4 points lower (95 % CI 2.9–5.1).", True),
             ("Higher-education plans", "Students who plan to go to university score ≈ 3.8 points higher (Cohen’s d ≈ 0.84).", True),
             ("Early grades", "The first-period grade G1 explains ≈ 64 % of the variation of the final grade.", True),
             ("Study time", "Only a weak link to the grade (ANOVA p ≈ 0.16, Spearman ρ ≈ 0.10).", False),
             ("Sex, internet, relationships", "Small effects that disappear after a Bonferroni correction.", False),
             ("Distributions", "Grades ≈ N(11.5, 3.2²) with clustering above the pass mark; absences are negative binomial (variance ≈ 11 × mean).", None)]
    for row in range(2):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            t_, d_, strong = items[row * 3 + j]
            tag = ('<span class="pill yes">strong effect</span>' if strong else
                   '<span class="pill no">weak / not robust</span>' if strong is False else
                   '<span class="pill yes">model fit</span>')
            col.markdown(f'<div class="card">{tag}<h4 style="margin-top:10px">{t_}</h4><p>{d_}</p></div>',
                         unsafe_allow_html=True)
        st.markdown("")
    st.markdown("### Limitations")
    st.markdown("Observational data (association ≠ causation) · two Portuguese schools in 2005–06 · self-reported "
                "variables · G3 = 0 most likely codes dropouts.")
    st.caption("Reference: Cortez, P. & Silva, A. (2008). Using Data Mining to Predict Secondary School Student "
               "Performance. UCI Machine Learning Repository.")
