"""What Drives Students' Final Math Grades? — interactive statistical study (Streamlit).
Run locally:  streamlit run app.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.optimize import minimize

st.set_page_config(page_title="Student Performance Study", page_icon="📊", layout="wide",
                   initial_sidebar_state="collapsed")

# ================================================================ design system
NAVY, SURFACE, SURFACE2 = "#0A1430", "#0F1C40", "#14244F"
TEXT, MUTED, LINE = "#F3F6FF", "#8E9BC4", "rgba(255,255,255,.08)"
BLUE, ORANGE = "#0297DE", "#E25A36"          # chart pair (validated on navy: lightness, contrast, colour-blindness)
ORANGE_UI = "#FC734D"                        # brand / buttons only
GRAY = "#3A4A78"
CMAP = LinearSegmentedColormap.from_list("bo", [BLUE, "#24345F", ORANGE])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": TEXT,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#2A3A68", "axes.labelcolor": MUTED, "axes.titlecolor": TEXT,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 10,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#1E2D5A", "grid.linewidth": .8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.frameon": False, "legend.labelcolor": TEXT, "patch.linewidth": 0, "lines.linewidth": 2,
})

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Playfair+Display:ital,wght@0,500;0,600;1,500;1,600&display=swap');
html, body, .stMarkdown, p, li, label, input, button, textarea {{ font-family: 'Inter', sans-serif; }}
.stApp {{ background: {NAVY}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {{ display: none; }}
.block-container {{ padding-top: 1.2rem; max-width: 1200px; }}
h1, h2, h3 {{ color: {TEXT}; letter-spacing: -0.01em; }}
h3 {{ font-family: 'Playfair Display', serif !important; font-weight: 600 !important; font-size: 1.7rem !important; }}

/* ---------- top bar ---------- */
.brand {{ font-weight: 700; font-size: 1rem; color: {TEXT}; display: flex; align-items: center; gap: 10px; padding-top: 6px; }}
.brand .bars {{ display: inline-flex; gap: 3px; align-items: flex-end; height: 14px; }}
.brand .bars i {{ display: block; width: 4px; background: {TEXT}; border-radius: 1px; }}
[data-testid="stPageLink"] a {{ border-radius: 999px; padding: 4px 12px; }}
[data-testid="stPageLink"] a p {{ color: {MUTED} !important; font-size: .86rem !important; font-weight: 500; }}
[data-testid="stPageLink"] a:hover p {{ color: {TEXT} !important; }}
.st-key-cta [data-testid="stPageLink"] a {{ background: {ORANGE_UI}; justify-content: center; }}
.st-key-cta [data-testid="stPageLink"] a p {{ color: white !important; font-weight: 600; }}
.topline {{ height: 1px; background: {LINE}; margin: 6px 0 22px; }}

/* ---------- landing hero ---------- */
.hero {{ position: relative; overflow: hidden; border-radius: 24px; background: {NAVY};
        border: 1px solid {LINE}; padding: 70px 40px 34px; min-height: 560px; }}
.hero .glow {{ position: absolute; left: 0; right: 0; bottom: 0; height: 78%;
        background:
          repeating-linear-gradient(90deg, rgba(10,20,48,1) 0 3px, transparent 3px 9px),
          linear-gradient(90deg, {BLUE} 0%, #3B5BDB 22%, {ORANGE_UI} 42%, #FF9A62 50%, {ORANGE_UI} 58%, #E0457B 72%, {BLUE} 100%);
        -webkit-mask-image: radial-gradient(ellipse 55% 85% at 50% 100%, black 0%, rgba(0,0,0,.55) 45%, transparent 75%),
                            linear-gradient(90deg, rgba(0,0,0,.45) 0 5%, transparent 16% 84%, rgba(0,0,0,.45) 95% 100%);
        -webkit-mask-composite: source-over; opacity: .9; }}
.hero .glow2 {{ position: absolute; inset: 0; background: radial-gradient(ellipse 40% 30% at 50% 100%, rgba(252,115,77,.35), transparent 70%); }}
.hero .content {{ position: relative; text-align: center; }}
.hero .kick {{ font-family: 'JetBrains Mono', monospace; font-size: .74rem; color: {MUTED}; letter-spacing: .14em; text-transform: uppercase; }}
.hero h1 {{ font-family: 'Playfair Display', serif !important; font-weight: 500; font-size: 4rem; line-height: 1.05;
        color: {TEXT}; margin: 16px 0 14px; padding: 0; letter-spacing: -0.02em; }}
.hero h1 em {{ font-style: italic; color: {TEXT}; }}
.hero .sub {{ font-family: 'JetBrains Mono', monospace; font-size: .82rem; color: #B8C3E6; line-height: 1.7; }}
.hero .stat {{ position: absolute; text-shadow: 0 2px 12px rgba(10,20,48,.9); font-family: 'Playfair Display', serif; color: {TEXT}; font-size: 1.6rem; }}
.hero .stat span {{ display: block; font-family: 'JetBrains Mono', monospace; font-size: .62rem; color: {MUTED}; letter-spacing: .06em; margin-top: 2px; }}
.hero .s1 {{ left: 44px; top: 300px; }} .hero .s2 {{ left: 44px; top: 380px; }}
.hero .s3 {{ right: 44px; top: 300px; text-align: right; }} .hero .s4 {{ right: 44px; top: 380px; text-align: right; }}
.panels {{ position: relative; display: grid; grid-template-columns: 1fr 1.15fr 1fr; gap: 16px; margin-top: 150px; }}
.panel {{ background: rgba(15,28,64,.88); border: 1px solid {LINE}; border-radius: 14px; padding: 12px 14px; backdrop-filter: blur(6px); text-align: left; }}
.panel.hot {{ background: linear-gradient(160deg, {ORANGE_UI}, #E0457B); border: none; }}
.panel .t {{ font-size: .72rem; color: #C9D2EE; margin-bottom: 10px; }}
.panel.hot .t {{ color: white; }}
@media (max-width: 900px) {{ .hero h1 {{ font-size: 2.6rem; }} .hero .stat {{ display: none; }} .panels {{ grid-template-columns: 1fr; margin-top: 60px; }} }}

/* ---------- section cards ---------- */
.sec-title {{ text-align: center; margin: 56px 0 6px; }}
.sec-title .kick {{ font-family: 'JetBrains Mono', monospace; font-size: .72rem; color: {ORANGE_UI}; letter-spacing: .14em; text-transform: uppercase; }}
.sec-title h2 {{ font-family: 'Playfair Display', serif !important; font-weight: 500; font-size: 2.6rem; margin: 8px 0 4px; }}
.sec-title h2 em {{ font-style: italic; }}
.sec-title p {{ color: {MUTED}; }}
[class*="st-key-card_"] {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 18px; padding: 20px 20px 10px;
        transition: transform .15s ease, border-color .15s ease; height: 100%; }}
[class*="st-key-card_"]:hover {{ transform: translateY(-3px); border-color: rgba(2,151,222,.55); }}
.card-num {{ font-family: 'JetBrains Mono', monospace; font-size: .72rem; color: {BLUE}; letter-spacing: .1em; }}
.card-h {{ font-family: 'Playfair Display', serif; font-size: 1.35rem; color: {TEXT}; margin: 6px 0 6px; line-height: 1.2; }}
.card-p {{ color: {MUTED}; font-size: .88rem; line-height: 1.5; min-height: 64px; }}
.card-viz {{ height: 54px; margin: 10px 0 4px; }}
[class*="st-key-card_"] [data-testid="stPageLink"] a {{ padding-left: 0; }}
[class*="st-key-card_"] [data-testid="stPageLink"] a p {{ color: {ORANGE_UI} !important; font-weight: 600; }}

/* ---------- inner pages ---------- */
.phead {{ position: relative; overflow: hidden; border-radius: 22px; border: 1px solid {LINE}; padding: 34px 38px 30px; margin-bottom: 26px;
        background: radial-gradient(ellipse 50% 120% at 100% 100%, rgba(252,115,77,.22), transparent 60%),
                    radial-gradient(ellipse 40% 100% at 0% 100%, rgba(2,151,222,.22), transparent 60%), {SURFACE}; }}
.phead .kick {{ font-family: 'JetBrains Mono', monospace; font-size: .72rem; color: {ORANGE_UI}; letter-spacing: .14em; text-transform: uppercase; }}
.phead h1 {{ font-family: 'Playfair Display', serif !important; font-weight: 500; font-size: 2.7rem; margin: 10px 0 8px; padding: 0; line-height: 1.1; }}
.phead h1 em {{ font-style: italic; }}
.phead p {{ color: #B8C3E6; margin: 0; max-width: 760px; line-height: 1.6; }}
[data-testid="stMetric"] {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 14px; padding: 16px 18px; }}
[data-testid="stMetricLabel"] p {{ color: {MUTED} !important; font-size: .8rem; }}
[data-testid="stMetricValue"] {{ font-family: 'Playfair Display', serif; color: {TEXT}; font-weight: 500; }}
.finding {{ border: 1px solid rgba(252,115,77,.35); background: rgba(252,115,77,.08); border-radius: 14px; padding: 14px 18px; margin: 10px 0 18px; color: #DDE3F7; }}
.finding b {{ color: {TEXT}; }}
.res {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 16px; padding: 18px 20px; height: 100%; }}
.res h4 {{ font-family: 'Playfair Display', serif; font-weight: 500; font-size: 1.25rem; margin: 10px 0 6px; color: {TEXT}; }}
.res p {{ color: {MUTED}; margin: 0; font-size: .9rem; line-height: 1.5; }}
.pill {{ display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: .7rem; font-weight: 600;
        font-family: 'JetBrains Mono', monospace; letter-spacing: .04em; }}
.pill.hot {{ background: rgba(252,115,77,.16); color: {ORANGE_UI}; }}
.pill.cool {{ background: rgba(2,151,222,.16); color: #4FC3F7; }}
.pill.dim {{ background: rgba(255,255,255,.06); color: {MUTED}; }}
[data-testid="stDataFrame"], [data-testid="stTable"] {{ border-radius: 12px; overflow: hidden; }}
.stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
.foot {{ text-align: center; color: {MUTED}; font-family: 'JetBrains Mono', monospace; font-size: .7rem; margin: 40px 0 10px; }}
</style>
""", unsafe_allow_html=True)


# ================================================================ data & helpers
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


def show(f):
    f.tight_layout()
    st.pyplot(f, width="stretch")
    plt.close(f)


def finding(html):
    st.markdown(f'<div class="finding">{html}</div>', unsafe_allow_html=True)


def page_header(kicker, title, text):
    st.markdown(f'<div class="phead"><div class="kick">{kicker}</div><h1>{title}</h1><p>{text}</p></div>',
                unsafe_allow_html=True)


# ================================================================ pages
def page_data():
    page_header("02 · Data", "Data &amp; <em>descriptive</em> statistics",
                "Two Portuguese secondary schools, 2005–06. G3 is the final grade (0–20); a student passes if G3 ≥ 10.")
    with st.expander("Show raw data"):
        st.dataframe(df, width="stretch")
    num = ["G1", "G2", "G3", "absences", "age", "studytime", "failures"]
    desc = df[num].agg(["mean", "median", "var", "std", "min", "max"]).T
    desc["skewness"] = df[num].apply(stats.skew)
    st.dataframe(desc.round(2), width="stretch")
    var = st.selectbox("Variable to plot", ["G3", "G1", "G2", "absences", "age", "studytime", "failures", "goout", "Walc"])
    f, ax = fig(10, 3.8)
    vals = df[var]
    ax.hist(vals, bins=np.arange(vals.min() - .5, vals.max() + 1.5, 1), color=BLUE, edgecolor=SURFACE, linewidth=1)
    ax.axvline(vals.mean(), color=ORANGE, ls="--", label=f"mean = {vals.mean():.2f}")
    ax.axvline(vals.median(), color=TEXT, ls=":", label=f"median = {vals.median():.1f}")
    ax.set(title=var, ylabel="students"); ax.legend(); show(f)
    st.caption("38 students have G3 = 0 while all have G1 > 0 — most likely they dropped out / missed the exam.")


def page_prob():
    page_header("03 · Probability", "Probability &amp; <em>Bayes’ theorem</em>",
                "Empirical and conditional probabilities, the law of total probability and Bayes’ theorem — computed live from the sample.")
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
    st.markdown(f"**Law of total probability:** {pAB:.3f}·{pB:.3f} + {pAnB:.3f}·{1 - pB:.3f} = "
                f"**{pAB * pB + pAnB * (1 - pB):.3f}** = P(pass) ✓")
    st.markdown(f"**Bayes:** P(B | not pass) = (1−{pAB:.3f})·{pB:.3f} / {1 - pA:.3f} = **{pB_notA:.3f}** (vs P(B) = {pB:.3f}).")
    indep = abs((A & B).mean() - pA * pB) < 0.005
    st.markdown(f"P(pass ∩ B) = {(A & B).mean():.3f}, P(pass)·P(B) = {pA * pB:.3f} → "
                + ("approximately **independent** in the sample." if indep else "**not independent** in the sample."))
    ct = pd.crosstab(B.map({True: factor, False: "not " + factor}), A.map({True: "pass", False: "fail"}), margins=True)
    st.dataframe(ct)


def page_est():
    page_header("04 · Estimation", "Method of moments <em>vs</em> maximum likelihood",
                "Which distribution fits grades and absences — and how MoM and MLE estimates compare.")
    tab1, tab2 = st.tabs(["Final grades — Normal", "Absences — Poisson vs Negative binomial"])
    with tab1:
        g = df.loc[df.G3 > 0, "G3"].values.astype(float)
        st.latex(r"\ell(\mu,\sigma)=-\tfrac n2\ln(2\pi\sigma^2)-\tfrac1{2\sigma^2}\sum(x_i-\mu)^2\ \Rightarrow\ "
                 r"\hat\mu=\bar x,\ \hat\sigma^2=\tfrac1n\sum(x_i-\bar x)^2")
        res = minimize(lambda t: -stats.norm.logpdf(g, t[0], np.exp(t[1])).sum(), [5, 0])
        mu, sd = res.x[0], np.exp(res.x[1])
        c = st.columns(3)
        c[0].metric("MoM μ̂ / σ̂", f"{g.mean():.3f} / {g.std():.3f}")
        c[1].metric("MLE μ̂ / σ̂", f"{mu:.3f} / {sd:.3f}")
        c[2].metric("n (G3 > 0)", len(g))
        f, ax = fig(10, 3.8); x = np.linspace(0, 21, 300)
        ax.hist(g, bins=np.arange(.5, 21.5, 1), density=True, color=BLUE, edgecolor=SURFACE, linewidth=1, label="data (G3 > 0)")
        ax.plot(x, stats.norm.pdf(x, mu, sd), color=ORANGE, lw=2.5, label=f"N({mu:.2f}, {sd:.2f}²)"); ax.legend(); show(f)
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
        ll_mom = stats.nbinom.logpmf(a, m**2 / (s2 - m), m / s2).sum()
        st.dataframe(pd.DataFrame({
            "model": ["Poisson (MLE)", "NegBin (MoM)", "NegBin (MLE)"],
            "parameters": [f"λ = {m:.3f}", f"r = {m**2 / (s2 - m):.3f}, p = {m / s2:.3f}", f"r = {r:.3f}, p = {p:.3f}"],
            "log-likelihood": [llp, ll_mom, -res.fun],
            "AIC": [2 - 2 * llp, 4 - 2 * ll_mom, 4 + 2 * res.fun]}).round(2), hide_index=True)
        k = np.arange(0, 31); f, ax = fig(10, 3.8)
        ax.bar(k, [np.mean(a == i) for i in k], color=GRAY, label="data")
        ax.plot(k, stats.poisson.pmf(k, m), "o-", color=BLUE, ms=5, label="Poisson")
        ax.plot(k, stats.nbinom.pmf(k, r, p), "s-", color=ORANGE, ms=5, label="Negative binomial"); ax.legend(); show(f)
        finding("<b>Negative binomial wins:</b> its AIC is about 2 000 lower than the Poisson model.")


def page_ci():
    page_header("05 · Inference", "Confidence <em>intervals</em>",
                "Interval estimates for the mean, the variance and the pass rate. Move the slider to change the confidence level.")
    conf = st.slider("Confidence level", .80, .99, .95, .01)
    z = stats.norm.ppf((1 + conf) / 2)
    var = st.selectbox("Variable for the mean", ["G3", "G1", "G2", "absences", "studytime"])
    x = df[var].astype(float); lo, hi = t_ci(x, conf)
    st.latex(r"\bar x \pm t_{\alpha/2,n-1}\frac{s}{\sqrt n}")
    c = st.columns(3)
    c[0].metric(f"{conf:.0%} CI · mean {var}", f"[{lo:.2f}, {hi:.2f}]")
    n = N; s2 = df.G3.var(ddof=1); al = 1 - conf
    c[1].metric(f"{conf:.0%} CI · variance of G3",
                f"[{(n - 1) * s2 / stats.chi2.ppf(1 - al / 2, n - 1):.1f}, {(n - 1) * s2 / stats.chi2.ppf(al / 2, n - 1):.1f}]")
    ph = df.passed.mean()
    c_ = (ph + z**2 / (2 * n)) / (1 + z**2 / n); h = z * np.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    c[2].metric(f"{conf:.0%} CI · pass rate (Wilson)", f"[{c_ - h:.3f}, {c_ + h:.3f}]")
    se = np.sqrt(ph * (1 - ph) / n)
    st.caption(f"Wald interval for the pass rate: [{ph - z * se:.3f}, {ph + z * se:.3f}]")

    st.markdown("### What does the confidence level mean?")
    m = st.slider("Sample size per interval", 10, 200, 40, 10)
    seed = st.number_input("Random seed", value=2026, step=1)
    rng = np.random.default_rng(int(seed)); pop = df.G3.values; mu = pop.mean(); cov = 0
    f, ax = fig(11, 4)
    for i in range(100):
        l, u = t_ci(rng.choice(pop, m), conf); ok = l <= mu <= u; cov += ok
        ax.plot([i, i], [l, u], color=BLUE if ok else ORANGE, lw=1.6 if ok else 2.6, solid_capstyle="round")
    ax.axhline(mu, color=TEXT, ls="--", lw=1)
    ax.set_title(f"{cov} of 100 intervals contain the true mean {mu:.2f}  (orange = missed)")
    show(f)


def page_tests():
    page_header("06 · Testing", "Hypothesis <em>tests</em>",
                "H₀: the factor is not associated with the final grade · H₁: it is (two-sided) · α = 0.05")
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
        c[0].metric(f"mean G3 · {a1} (n={len(x1)})", f"{x1.mean():.2f}")
        c[1].metric(f"mean G3 · {a2} (n={len(x2)})", f"{x2.mean():.2f}")
        c[2].metric("Cohen's d", f"{cohen_d(x1, x2):.2f}")
        c[3].metric("Welch t", f"{t.statistic:.2f}")
        tbl = pd.DataFrame({"test": ["Welch t-test", "Mann–Whitney U", "χ² independence (vs pass/fail)"],
                            "p-value": [t.pvalue, u.pvalue, chi[1]]})
    else:
        groups = [v.G3.values for _, v in df.groupby(col)]
        f_ = stats.f_oneway(*groups); kw = stats.kruskal(*groups)
        chi = stats.chi2_contingency(pd.crosstab(df[col], df.passed))
        st.metric("ANOVA F", f"{f_.statistic:.2f}")
        tbl = pd.DataFrame({"test": ["One-way ANOVA", "Kruskal–Wallis", "χ² independence (vs pass/fail)"],
                            "p-value": [f_.pvalue, kw.pvalue, chi[1]]})
    tbl["decision"] = np.where(tbl["p-value"] < .05, "Reject H₀", "Fail to reject H₀")
    tbl["Bonferroni (α/10)"] = np.where(tbl["p-value"] < .005, "Reject H₀", "Fail to reject H₀")
    st.dataframe(tbl.style.format({"p-value": "{:.2e}"}), hide_index=True, width="stretch")
    grp = df.groupby(col).G3
    means = grp.mean(); err = grp.apply(lambda x: x.mean() - t_ci(x)[0] if len(x) > 1 else 0)
    f, axs = fig(12, 3.8, 2)
    axs[0].bar(means.index.astype(str), means, yerr=err, capsize=5, color=BLUE, ecolor=TEXT)
    axs[0].set_title("Mean G3 ± 95 % CI")
    pr = df.groupby(col).passed.mean(); axs[1].bar(pr.index.astype(str), pr, color=ORANGE)
    axs[1].set_title("Pass rate"); axs[1].set_ylim(0, 1); show(f)
    st.caption("We run many tests, so some 'significant' results may be false positives. The Bonferroni column uses α = 0.005.")


def page_corr():
    page_header("07 · Relationships", "Correlation &amp; <em>regression</em>",
                "How strongly do early grades and habits move together with the final grade?")
    cols = ["G1", "G2", "G3", "failures", "studytime", "absences", "goout", "Walc", "age", "Medu"]
    method = st.radio("Correlation", ["spearman", "pearson"], horizontal=True)
    corr = df[cols].corr(method=method)
    f, ax = fig(7.5, 6); im = ax.imshow(corr, cmap=CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45); ax.set_yticks(range(len(cols)), cols); ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7, color=TEXT)
    cb = plt.colorbar(im); cb.outline.set_visible(False); show(f)
    xv = st.selectbox("Predictor of G3", ["G1", "G2", "failures", "absences", "studytime", "goout", "age"])
    lr = stats.linregress(df[xv], df.G3)
    st.latex(rf"\widehat{{G3}} = {lr.intercept:.3f} {'+' if lr.slope >= 0 else '-'} {abs(lr.slope):.3f}\cdot {xv}"
             rf"\qquad r={lr.rvalue:.3f},\ R^2={lr.rvalue**2:.3f},\ p={lr.pvalue:.1e}")
    rng = np.random.default_rng(1); j = rng.uniform(-.25, .25, N)
    f, ax = fig(9, 4); ax.scatter(df[xv] + j, df.G3 + j, s=16, alpha=.55, color=BLUE, edgecolors="none")
    xx = np.linspace(df[xv].min(), df[xv].max(), 10); ax.plot(xx, lr.intercept + lr.slope * xx, color=ORANGE, lw=2.5)
    ax.set(xlabel=xv, ylabel="G3"); show(f)


def page_clt():
    page_header("08 · Simulation", "Central Limit <em>Theorem</em>",
                "Draw thousands of samples from a skewed variable and watch the sample means become normal.")
    st.latex(r"\bar X_n \;\approx\; N\!\left(\mu,\ \frac{\sigma^2}{n}\right)\quad\text{for large } n")
    var = st.selectbox("Population (our data)", ["absences", "G3", "failures", "Walc"])
    n = st.slider("Sample size n", 1, 200, 30)
    reps = st.select_slider("Number of samples", [1000, 5000, 10000, 20000], 10000)
    pop = df[var].values.astype(float); rng = np.random.default_rng(0)
    means = rng.choice(pop, (reps, n)).mean(axis=1)
    f, axs = fig(13, 4, 2)
    axs[0].hist(pop, bins=30, color=GRAY, edgecolor=SURFACE); axs[0].set_title(f"Population · skew = {stats.skew(pop):.2f}")
    axs[1].hist(means, bins=50, density=True, color=BLUE, edgecolor=SURFACE, linewidth=.5)
    xs = np.linspace(means.min(), means.max(), 200)
    axs[1].plot(xs, stats.norm.pdf(xs, pop.mean(), pop.std() / np.sqrt(n)), color=ORANGE, lw=2.5, label="CLT normal")
    axs[1].set_title(f"Sample means, n = {n} · skew = {stats.skew(means):.2f}"); axs[1].legend(); show(f)
    c = st.columns(3)
    c[0].metric("SD of sample means", f"{means.std():.3f}")
    c[1].metric("Theory σ/√n", f"{pop.std() / np.sqrt(n):.3f}")
    c[2].metric("Mean of sample means", f"{means.mean():.3f}", f"μ = {pop.mean():.3f}", delta_color="off")


def page_concl():
    page_header("09 · Results", "What the data <em>say</em>",
                "Main findings of the study — and what an observational dataset cannot tell us.")
    items = [("Past failures", "P(pass | ≥1 failure) = 0.37 vs 0.75 without; mean grade ≈ 4 points lower (95 % CI 2.9–5.1).", "hot", "strong effect"),
             ("Higher-education plans", "Students who plan to go to university score ≈ 3.8 points higher (Cohen’s d ≈ 0.84).", "hot", "strong effect"),
             ("Early grades", "The first-period grade G1 explains ≈ 64 % of the variation of the final grade.", "hot", "strong effect"),
             ("Study time", "Only a weak link to the grade (ANOVA p ≈ 0.16, Spearman ρ ≈ 0.10).", "dim", "weak"),
             ("Sex, internet, relationships", "Small effects that disappear after a Bonferroni correction.", "dim", "not robust"),
             ("Distributions", "Grades ≈ N(11.5, 3.2²) with clustering above the pass mark; absences are negative binomial (variance ≈ 11 × mean).", "cool", "model fit")]
    for row in range(2):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            t_, d_, cls, tag = items[row * 3 + j]
            col.markdown(f'<div class="res"><span class="pill {cls}">{tag}</span><h4>{t_}</h4><p>{d_}</p></div>',
                         unsafe_allow_html=True)
        st.markdown("")
    st.markdown("### Limitations")
    st.markdown("Observational data (association ≠ causation) · two Portuguese schools in 2005–06 · self-reported "
                "variables · G3 = 0 most likely codes dropouts.")
    st.caption("Reference: Cortez, P. & Silva, A. (2008). Using Data Mining to Predict Secondary School Student "
               "Performance. UCI Machine Learning Repository.")


# ---------------------------------------------------------------- mini visuals for the landing cards (inline SVG)
def svg_bars(vals, color):
    w = 100 / len(vals)
    bars = "".join(f'<rect x="{i * w + 1}" y="{54 - v * 50}" width="{w - 2.5}" height="{v * 50}" rx="1.5" fill="{color}"/>'
                   for i, v in enumerate(vals))
    return f'<svg class="card-viz" viewBox="0 0 100 54" preserveAspectRatio="none" width="100%">{bars}</svg>'


def svg_curve(color, pts):
    path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))
    return (f'<svg class="card-viz" viewBox="0 0 100 54" preserveAspectRatio="none" width="100%">'
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.2" vector-effect="non-scaling-stroke"/></svg>')


_xs = np.linspace(0, 100, 40)
NORMAL_PTS = [(x, 52 - 46 * np.exp(-((x - 50) / 16) ** 2)) for x in _xs]
SKEW_PTS = [(x, 52 - 48 * (x / 18) * np.exp(1 - x / 18)) for x in _xs]
LINE_PTS = [(x, 48 - x * .42 + 4 * np.sin(x / 7)) for x in _xs]


def home():
    st.markdown(f"""
<div class="hero">
  <div class="glow"></div><div class="glow2"></div>
  <div class="content">
    <div class="kick">Research study · n = {N} · Probability &amp; Statistics</div>
    <h1>What drives students’<br><em>final grades?</em></h1>
    <div class="sub">Probability, estimation, confidence intervals and hypothesis tests<br>on real data from {N} secondary-school students.</div>
  </div>
  <div class="stat s1">{N}<span>students observed</span></div>
  <div class="stat s2">{df.passed.mean():.0%}<span>pass rate (G3 ≥ 10)</span></div>
  <div class="stat s3">{df.G3.mean():.2f}<span>mean final grade / 20</span></div>
  <div class="stat s4">{df.shape[1] - 1}<span>variables per student</span></div>
  <div class="panels">
    <div class="panel"><div class="t">Pass rate by past failures</div>
      {svg_bars([df.loc[df.failures == k, 'passed'].mean() for k in range(4)], BLUE)}</div>
    <div class="panel hot"><div class="t">Final grades · normal fit</div>{svg_curve('white', NORMAL_PTS)}</div>
    <div class="panel"><div class="t">Absences · negative binomial</div>{svg_curve(ORANGE_UI, SKEW_PTS)}</div>
  </div>
</div>
<div class="sec-title"><div class="kick">Explore the study</div><h2>Eight steps, <em>one question</em></h2>
<p>Pick a section — each one opens on its own page.</p></div>
""", unsafe_allow_html=True)

    g3_hist = np.histogram(df.G3, bins=10)[0]
    cards = [
        (P_DATA, "02", "Data &amp; descriptive stats", "Means, variances, skewness and the shape of every variable.",
         svg_bars(list(g3_hist / g3_hist.max()), BLUE)),
        (P_PROB, "03", "Probability &amp; Bayes", "Conditional probability, total probability and Bayes’ theorem.",
         svg_bars([.67, .75, .37, .40], ORANGE_UI)),
        (P_EST, "04", "MoM &amp; MLE", "Fitting normal, Poisson and negative binomial models.", svg_curve(BLUE, NORMAL_PTS)),
        (P_CI, "05", "Confidence intervals", "Interval estimates and what “95 %” really means.",
         svg_bars([.5, .62, .44, .7, .55, .66, .48, .6, .52, .68], BLUE)),
        (P_TESTS, "06", "Hypothesis tests", "t-tests, ANOVA, Mann–Whitney and χ² for 16 factors.",
         svg_bars([11.25 / 12, 8.12 / 12, 6.24 / 12, 5.69 / 12], ORANGE_UI)),
        (P_CORR, "07", "Correlation &amp; regression", "How early grades predict the final one.", svg_curve(BLUE, LINE_PTS)),
        (P_CLT, "08", "CLT simulator", "Watch skewed data produce normal sample means.", svg_curve(ORANGE_UI, NORMAL_PTS)),
        (P_CONCL, "09", "Conclusions", "Strong effects, weak effects and limitations.",
         svg_bars([.95, .85, .8, .3, .25, .2], BLUE)),
    ]
    for row in range(0, len(cards), 4):
        cols = st.columns(4, gap="medium")
        for col, (pg, num, title, text, viz) in zip(cols, cards[row:row + 4]):
            with col:
                with st.container(key=f"card_{num}"):
                    st.markdown(f'<div class="card-num">{num}</div><div class="card-h">{title}</div>'
                                f'<div class="card-p">{text}</div>{viz}', unsafe_allow_html=True)
                    st.page_link(pg, label="Open section →")
        st.markdown("")
    st.markdown('<div class="foot">Data: Cortez &amp; Silva (2008), UCI Student Performance · Narxoz University</div>',
                unsafe_allow_html=True)


# ================================================================ navigation
P_HOME = st.Page(home, title="Overview", url_path="home", default=True)
P_DATA = st.Page(page_data, title="Data", url_path="data")
P_PROB = st.Page(page_prob, title="Probability", url_path="probability")
P_EST = st.Page(page_est, title="Estimation", url_path="estimation")
P_CI = st.Page(page_ci, title="Intervals", url_path="intervals")
P_TESTS = st.Page(page_tests, title="Tests", url_path="tests")
P_CORR = st.Page(page_corr, title="Regression", url_path="regression")
P_CLT = st.Page(page_clt, title="CLT", url_path="clt")
P_CONCL = st.Page(page_concl, title="Conclusions", url_path="conclusions")
ORDER = [P_HOME, P_DATA, P_PROB, P_EST, P_CI, P_TESTS, P_CORR, P_CLT, P_CONCL]

current = st.navigation(ORDER, position="hidden")
ON_HOME = current.title == "Overview"

# top bar (every page)
bar = st.columns([2.2, .9, .9, .9, .9, .9, 1.3], vertical_alignment="center")
bar[0].markdown('<div class="brand"><span class="bars"><i style="height:8px"></i><i style="height:14px"></i>'
                '<i style="height:11px"></i></span>Student Performance</div>', unsafe_allow_html=True)
for col, pg, label in zip(bar[1:6], [P_HOME, P_PROB, P_EST, P_TESTS, P_CLT],
                          ["Overview", "Probability", "Estimation", "Tests", "CLT"]):
    with col:
        st.page_link(pg, label=label)
with bar[6]:
    with st.container(key="cta"):
        st.page_link(P_CONCL if ON_HOME else P_HOME, label="See results →" if ON_HOME else "← Overview")
st.markdown('<div class="topline"></div>', unsafe_allow_html=True)

current.run()

# previous / next links on section pages
if not ON_HOME:
    i = [p.title for p in ORDER].index(current.title)
    st.markdown('<div class="topline" style="margin-top:34px"></div>', unsafe_allow_html=True)
    nav = st.columns([1, 1, 1])
    with nav[0]:
        st.page_link(ORDER[i - 1], label=f"← {ORDER[i - 1].title}")
    with nav[2]:
        if i + 1 < len(ORDER):
            st.page_link(ORDER[i + 1], label=f"{ORDER[i + 1].title} →")
