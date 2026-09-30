"""ESG 환경등급과 온실가스 배출 성과 분석 대시보드 (pj3)

실행:  uv run streamlit run app.py
구성:  소개 3페이지 + 분석 5페이지 + 부록 1페이지 (사이드바 / 이전·다음 버튼으로 이동)

데이터 (./data):
  sbti.csv, kcgs.csv, gir_company_2020_2025_reviewed.csv
  dashboard_primary.csv       ← 노트북(analysis4_reviewed)에서 내보낸 t-1 결합 결과 (있으면 주 분석으로 사용)
  dashboard_same_year.csv     ← 같은 노트북의 동일연도 결합 결과 (선택, 민감도 분석용)
  esg_environment_final.csv   ← 위 두 파일이 없을 때 쓰는 임시 데이터(동일연도 정렬)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import spearmanr

PROJECT_TITLE = "박준형의 개인프로젝트 2"
PROJECT_SUBTITLE = "ESG 환경등급과 온실가스 배출 성과 분석"

st.set_page_config(
    page_title=f"{PROJECT_TITLE} | {PROJECT_SUBTITLE}",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# 상수 · 색상 · 스타일
# ──────────────────────────────────────────────────────────────
DATA = Path(__file__).parent / "data"

GRADES = ["A+", "A", "B+", "B", "C", "D"]
GRADE_RANK = {"D": 1, "C": 2, "B": 3, "B+": 4, "A": 5, "A+": 6}
EMIS = "온실가스 배출량(tCO₂eq)"
IND = "지정업종(목표)/<br />계획업종(할당)"
KOREA = "Korea, Republic of"
STATUS_ORDER = ["Targets set", "Committed", "Commitment removed", "미기재"]
PERIODS = [(2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]

DEEP = "#0B3D2E"      # 브랜드 딥그린 (제목·사이드바)
INK = "#14261E"
SUB = "#4A5F55"
MUTED = "#7A8C83"
GRID = "#E3ECE7"

# 범주 색: 초록·주황·파랑 (palette validator 통과 조합, 주황은 직접 라벨로 보완)
C_GREEN = "#12855A"
C_AMBER = "#D9822B"
C_BLUE = "#3B78C2"
C_GRAY = "#B8C2BC"     # 중립(미기재·기준 집단)
# 등급: 딥그린 한 색의 명도 단계 (A+ 진함 → D 연함)
GRADE_COLOR = {
    "A+": "#0B3D2E", "A": "#17634A", "B+": "#2F8F68",
    "B": "#6DB593", "C": "#A9D3BC", "D": "#D3E8DB",
}
STATUS_COLOR = {
    "Targets set": C_GREEN, "Committed": C_BLUE,
    "Commitment removed": C_AMBER, "미기재": C_GRAY,
}
CHANGE_COLOR = {"상향": C_GREEN, "유지": C_GRAY, "하향": C_AMBER}
FONT = "Pretendard, 'Noto Sans KR', 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif"

CSS = f"""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css');
html, body {{ font-family: {FONT}; }}
.stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp td, .stApp th, .stApp button, .stApp input, .stApp textarea {{ font-family: {FONT}; }}
.stApp {{ background: #F7FAF8; color: {INK}; }}
.block-container {{ padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1180px; }}
h1, h2, h3, h4 {{ color: {DEEP}; letter-spacing: -0.01em; }}

section[data-testid="stSidebar"] {{ background: {DEEP}; }}
section[data-testid="stSidebar"] * {{ color: #E4F1E9 !important; }}
section[data-testid="stSidebar"] a {{ border-radius: 8px; }}
section[data-testid="stSidebar"] a:hover {{ background: rgba(255,255,255,0.08); }}
section[data-testid="stSidebar"] a[aria-current="page"] {{ background: rgba(255,255,255,0.16); }}

.kicker {{ color: {C_GREEN}; font-weight: 700; font-size: .85rem; letter-spacing: .05em; }}
.brand {{ display: flex; align-items: center; gap: 8px; color: {MUTED}; font-size: .8rem; margin-bottom: 1.1rem; }}
.brand b {{ color: {DEEP}; }}
.brand .dot {{ width: 8px; height: 8px; border-radius: 50%; background: {C_GREEN}; display: inline-block; }}
.hero {{ background: linear-gradient(135deg, #0B3D2E 0%, #12855A 100%); border-radius: 20px;
         padding: 34px 38px; margin-bottom: 1.6rem; }}
.hero .eyebrow {{ color: #A9D3BC; font-size: .85rem; font-weight: 700; letter-spacing: .08em; }}
.hero .title {{ color: #FFFFFF; font-size: 2.5rem; font-weight: 800; line-height: 1.2; margin: .3rem 0 .4rem; }}
.hero .subtitle {{ color: #E4F1E9; font-size: 1.25rem; font-weight: 500; }}
.hero .tags {{ margin-top: 16px; }}
.hero .tag {{ display: inline-block; background: rgba(255,255,255,.14); color: #E4F1E9; border-radius: 999px;
              padding: 3px 12px; margin-right: 8px; font-size: .8rem; }}
.page-title {{ color: {DEEP}; font-size: 2.05rem; font-weight: 800; line-height: 1.25; margin: .15rem 0 .5rem; }}
.question {{ color: {SUB}; font-size: 1.05rem; margin-bottom: 1.4rem; line-height: 1.6; }}
.kpi {{ background: #fff; border: 1px solid #DCE8E1; border-radius: 14px; padding: 16px 18px; height: 100%; }}
.kpi .label {{ color: #5B6F65; font-size: .82rem; }}
.kpi .value {{ color: {DEEP}; font-size: 1.85rem; font-weight: 800; line-height: 1.25; }}
.kpi .note {{ color: {MUTED}; font-size: .78rem; margin-top: 2px; }}
.card {{ background: #fff; border: 1px solid #DCE8E1; border-radius: 14px; padding: 18px 20px; height: 100%; }}
.card h4 {{ margin: 0 0 6px; font-size: 1.05rem; }}
.card p {{ margin: 0; color: {SUB}; font-size: .93rem; line-height: 1.6; }}
.step-no {{ display: inline-block; background: {DEEP}; color: #fff; border-radius: 999px;
            padding: 2px 10px; font-size: .78rem; font-weight: 700; margin-bottom: 8px; }}
.step-result {{ margin-top: 10px; padding-top: 8px; border-top: 1px dashed #CFE0D6;
                color: {C_GREEN}; font-size: .86rem; font-weight: 600; }}
.callout {{ border-left: 4px solid {C_GREEN}; background: #EAF4EE; padding: 12px 16px;
            border-radius: 0 10px 10px 0; color: {INK}; margin: .6rem 0 1rem; line-height: 1.65; }}
.caveat {{ border-left: 4px solid {C_AMBER}; background: #FDF3E7; padding: 12px 16px;
           border-radius: 0 10px 10px 0; color: {INK}; margin: .6rem 0 1rem; line-height: 1.65; }}
.body-text {{ font-size: 1.02rem; line-height: 1.85; color: {INK}; }}
.quote {{ font-size: 1.25rem; font-weight: 700; color: {DEEP}; border-left: 5px solid {C_GREEN};
          padding: 6px 0 6px 16px; margin: 1.1rem 0; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# UI 헬퍼
# ──────────────────────────────────────────────────────────────
def header(kicker: str, title: str, question: str = "", banner: bool = True) -> None:
    if banner:
        st.markdown(f'<div class="brand"><span class="dot"></span><b>{PROJECT_TITLE}</b>'
                    f'<span>·</span><span>{PROJECT_SUBTITLE}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kicker">{kicker}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    if question:
        st.markdown(f'<div class="question">{question}</div>', unsafe_allow_html=True)


def kpi(col, label: str, value: str, note: str = "") -> None:
    col.markdown(
        f'<div class="kpi"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def callout(text: str) -> None:
    st.markdown(f'<div class="callout">{text}</div>', unsafe_allow_html=True)


def caveat(text: str) -> None:
    st.markdown(f'<div class="caveat"><b>해석 시 주의</b><br>{text}</div>', unsafe_allow_html=True)


def body(text: str) -> None:
    st.markdown(f'<div class="body-text">{text}</div>', unsafe_allow_html=True)


def plot(fig: go.Figure, key: str) -> None:
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:  # 구버전 Streamlit
        st.plotly_chart(fig, use_container_width=True, key=key)


def table(df: pd.DataFrame, height: int | None = None) -> None:
    kw = {"height": height} if height else {}
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


def style_fig(fig: go.Figure, title: str = "", height: int = 380, horizontal: bool = False) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        title=dict(text=title, x=0, font=dict(size=16, color=INK)),
        font=dict(family=FONT, color=INK, size=13),
        margin=dict(l=8, r=8, t=56 if title else 24, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title_text=""),
        hoverlabel=dict(bgcolor="white", font_size=13),
    )
    if horizontal:  # 가로 막대: 값 축(x)에만 격자
        fig.update_xaxes(showgrid=True, gridcolor=GRID, linecolor=GRID, zeroline=False)
        fig.update_yaxes(showgrid=False, linecolor=GRID)
    else:
        fig.update_xaxes(showgrid=False, linecolor=GRID, tickcolor=GRID)
        fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def nav_footer(i: int) -> None:
    st.divider()
    left, mid, right = st.columns([1, 1, 1])
    if i > 0:
        left.page_link(ORDER[i - 1], label=f"← {ORDER[i - 1].title}")
    if i < len(ORDER) - 1:
        right.page_link(ORDER[i + 1], label=f"{ORDER[i + 1].title} →")


def pct(v: float, digits: int = 1) -> str:
    return "–" if pd.isna(v) else f"{v:.{digits}f}%"


# ──────────────────────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_sbti() -> pd.DataFrame:
    cols = ["company_name", "location", "region", "sector", "organization_type",
            "near_term_status", "near_term_target_classification", "net_zero_status"]
    df = pd.read_csv(DATA / "sbti.csv", usecols=cols, encoding="utf-8-sig")
    df["near_term_status"] = df["near_term_status"].fillna("미기재")
    df["net_zero_status"] = df["net_zero_status"].fillna("미기재")
    return df


@st.cache_data(show_spinner=False)
def load_kcgs() -> pd.DataFrame:
    df = pd.read_csv(DATA / "kcgs.csv", dtype={"기업코드": str}, encoding="utf-8-sig")
    df = df[df["평가년도"].between(2020, 2025)].copy()
    df["평가년도"] = df["평가년도"].astype(int)
    return df[df["환경"].isin(GRADES)].copy()


@st.cache_data(show_spinner=False)
def load_gir() -> pd.DataFrame:
    df = pd.read_csv(DATA / "gir_company_2020_2025_reviewed.csv", encoding="utf-8-sig")
    df = df[df["대상년도"].between(2020, 2025)].copy()
    df["대상년도"] = df["대상년도"].astype(int)
    return df


COMBINED_COLS = ["기업키", "기업명", "평가년도", "성과연도", "환경", "환경등급순위", EMIS, "배출집약도", IND]


@st.cache_data(show_spinner=False)
def load_combined() -> tuple[dict, bool]:
    """(정렬 기준 이름 → 결합 데이터, 임시 데이터 여부)"""
    frames: dict[str, pd.DataFrame] = {}
    files = [("t-1 (주 분석)", "dashboard_primary.csv"), ("동일연도 (민감도)", "dashboard_same_year.csv")]
    for label, name in files:
        path = DATA / name
        if path.exists():
            frames[label] = pd.read_csv(path, encoding="utf-8-sig")
    if frames:
        return {k: _clean_combined(v) for k, v in frames.items()}, False

    # 임시: 기존 결합 파일(동일연도 정렬)
    d = pd.read_csv(DATA / "esg_environment_final.csv", dtype={"종목코드_clean": str}, encoding="utf-8-sig")
    d = d[d["환경"].isin(GRADES)].copy()
    d["기업키"] = np.where(d["종목코드_clean"].notna(), "C:" + d["종목코드_clean"].fillna(""), "N:" + d["기업명_norm"])
    d["평가년도"] = d["평가년도"].astype(int)
    d["성과연도"] = d["연도"].astype(int)
    return {"동일연도 (임시 데이터)": _clean_combined(d[COMBINED_COLS])}, True


def _clean_combined(df: pd.DataFrame) -> pd.DataFrame:
    out = df[COMBINED_COLS].copy()
    out["평가년도"] = out["평가년도"].astype(int)
    out["환경등급순위"] = out["환경"].map(GRADE_RANK)
    out = out[out["환경"].isin(GRADES)]
    return out.drop_duplicates(["기업키", "평가년도"]).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
# 계산 함수 (노트북 로직과 동일)
# ──────────────────────────────────────────────────────────────
def share(series: pd.Series, order=STATUS_ORDER) -> pd.Series:
    return series.value_counts(normalize=True).mul(100).reindex(order).fillna(0)


def safe_spearman(df: pd.DataFrame, x: str, y: str):
    t = df[[x, y]].dropna()
    if len(t) < 2 or t[x].nunique() < 2 or t[y].nunique() < 2:
        return np.nan, np.nan, len(t)
    rho, p = spearmanr(t[x], t[y])
    return float(rho), float(p), len(t)


def kcgs_change(k: pd.DataFrame, y0: int, y1: int) -> pd.DataFrame:
    a = k[k["평가년도"] == y0][["기업코드", "환경"]]
    b = k[k["평가년도"] == y1][["기업코드", "환경"]]
    m = a.merge(b, on="기업코드", suffixes=(f"_{y0}", f"_{y1}"))
    diff = m[f"환경_{y1}"].map(GRADE_RANK) - m[f"환경_{y0}"].map(GRADE_RANK)
    m["등급변화"] = np.select([diff > 0, diff < 0], ["상향", "하향"], default="유지")
    return m


def make_changes(frame: pd.DataFrame) -> pd.DataFrame:
    c = frame.sort_values(["기업키", "평가년도"]).copy()
    g = c.groupby("기업키", sort=False)
    c["전년도"] = g["평가년도"].shift(1)
    c["전년등급순위"] = g["환경등급순위"].shift(1)
    c["전년집약도"] = g["배출집약도"].shift(1)
    c = c[(c["평가년도"] - c["전년도"]).eq(1)].copy()
    c["등급변화폭"] = c["환경등급순위"] - c["전년등급순위"]
    c["등급변화구분"] = np.select([c["등급변화폭"] > 0, c["등급변화폭"] < 0], ["상향", "하향"], default="유지")
    c["집약도증감률"] = np.where(
        c["전년집약도"] > 0, (c["배출집약도"] - c["전년집약도"]) / c["전년집약도"] * 100, np.nan
    )
    return c


def change_table(change: pd.DataFrame) -> pd.DataFrame:
    v = change.dropna(subset=["집약도증감률"])
    return (
        v.groupby("등급변화구분")
        .agg(유효비교수=("기업키", "size"), 고유기업수=("기업키", "nunique"), 집약도증감률_중앙값=("집약도증감률", "median"))
        .reindex(["상향", "유지", "하향"])
    )


def active_frame(frames: dict) -> tuple[str, pd.DataFrame]:
    label = next(iter(frames))
    return label, frames[label]


# ──────────────────────────────────────────────────────────────
# 소개 1. ESG란
# ──────────────────────────────────────────────────────────────
def page_esg():
    st.markdown(
        f'<div class="hero"><div class="eyebrow">PERSONAL PROJECT 2</div>'
        f'<div class="title">{PROJECT_TITLE}</div><div class="subtitle">{PROJECT_SUBTITLE}</div>'
        '<div class="tags"><span class="tag">SBTi</span><span class="tag">KCGS</span>'
        '<span class="tag">GIR</span><span class="tag">DART</span><span class="tag">Python · Pandas · Plotly · Streamlit</span></div></div>',
        unsafe_allow_html=True,
    )
    header("소개 1", "ESG란", "기업을 재무성과만이 아니라 환경·사회·지배구조까지 함께 보는 관점입니다.", banner=False)
    body(
        "ESG는 기업을 평가할 때 재무성과뿐 아니라 환경(Environment), 사회(Social), 지배구조(Governance) "
        "측면의 지속가능성을 함께 보는 개념이다."
    )
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="card"><h4>🌱 환경 (E)</h4><p>온실가스 배출, 에너지 사용, 폐기물·수질 관리처럼 '
                "기업 활동이 환경에 미치는 영향을 다룬다.</p></div>", unsafe_allow_html=True)
    c2.markdown('<div class="card"><h4>🤝 사회 (S)</h4><p>노동·안전, 인권, 지역사회처럼 '
                "사람과 관련된 기업의 책임을 다룬다.</p></div>", unsafe_allow_html=True)
    c3.markdown('<div class="card"><h4>🏛️ 지배구조 (G)</h4><p>이사회 구성, 경영 투명성, 주주 권리처럼 '
                "기업이 운영되는 방식을 다룬다.</p></div>", unsafe_allow_html=True)
    st.write("")
    body("이 프로젝트에서는 그중 <b>환경(E), 특히 온실가스</b>에 집중했다. 기업의 환경 대응을 다음 데이터로 나누어 살펴봤다.")
    st.write("")
    a, b = st.columns(2)
    a.markdown('<div class="card"><h4>SBTi</h4><p>기업이 과학적 기준에 맞는 온실가스 감축목표를 설정했는지 확인</p></div>', unsafe_allow_html=True)
    b.markdown('<div class="card"><h4>KCGS</h4><p>국내 기업 ESG 평가 중 환경등급 확인</p></div>', unsafe_allow_html=True)
    st.write("")
    a, b = st.columns(2)
    a.markdown('<div class="card"><h4>GIR</h4><p>국내 관리·할당 대상 기업의 온실가스 배출량 확인</p></div>', unsafe_allow_html=True)
    b.markdown('<div class="card"><h4>DART · 배출집약도</h4><p>DART 매출액으로 기업 규모의 영향을 줄이기 위해 '
               "매출액 10억원당 온실가스 배출량(배출집약도)을 계산</p></div>", unsafe_allow_html=True)
    st.write("")
    callout("즉, 기업이 어떤 목표를 세웠는지, 어떤 환경등급을 받았는지, 실제 온실가스 배출 성과는 어떠한지를 "
            "나누어 살펴보고, 마지막에 <b>환경등급과 배출 성과의 관계</b>를 확인했다.")
    nav_footer(0)


# ──────────────────────────────────────────────────────────────
# 소개 2. 이 프로젝트를 하는 이유
# ──────────────────────────────────────────────────────────────
def page_why():
    header("소개 2", "이 프로젝트를 하는 이유", "같은 환경 데이터도 어떻게 정리하고 비교하느냐에 따라 다르게 보입니다.")
    body(
        "환경공학을 전공하면서 환경 문제를 주로 이론과 수치로 배웠지만, 데이터 분석을 배우기 시작하면서 "
        "같은 환경 문제도 데이터를 어떻게 정리하고 비교하느냐에 따라 다르게 볼 수 있다는 점에 흥미를 느꼈다."
        "<br><br>"
        "특히 국가 온실가스 인벤토리 데이터를 활용해 대시보드를 만들어보면서 단순히 배출량 하나를 보는 것보다 "
        "연도별 변화, 분야별 비중, 증감률과 같은 지표를 함께 봐야 전체 흐름을 이해할 수 있다는 것을 경험했다."
        "<br><br>그러면서 기업 ESG에도 비슷한 의문이 생겼다."
    )
    st.markdown('<div class="quote">“ESG 환경등급이 높은 기업은 실제 온실가스 배출 성과도 좋은 기업일까?”</div>',
                unsafe_allow_html=True)
    body(
        "처음에는 환경등급과 배출량을 단순 비교하면 답을 얻을 수 있을 것이라고 생각했다. 하지만 기업마다 규모와 "
        "업종이 다르기 때문에 절대배출량만으로 환경성과를 비교하는 것은 한계가 있다고 판단했다."
        "<br><br>"
        "그래서 이번 프로젝트에서는 SBTi로 기업들의 기후목표 설정 현황을 먼저 살펴보고, KCGS 환경등급·GIR 온실가스 "
        "배출량·DART 매출액을 연결해 평가등급과 실제 배출 성과가 어떤 관계를 가지는지 단계적으로 확인하고자 했다."
        "<br><br>"
        "이 프로젝트를 통해 환경공학에서 배운 내용을 데이터 분석과 연결하고, 앞으로 관심을 가지고 있는 "
        "ESG 경영, 환경데이터 관리, 온실가스 데이터 관리 업무에서 실제로 데이터를 어떻게 검증하고 활용할 수 있을지 "
        "경험해보는 것이 가장 큰 목적이었다."
    )
    nav_footer(1)


# ──────────────────────────────────────────────────────────────
# 소개 3. 프로젝트 개요 및 순서
# ──────────────────────────────────────────────────────────────
def page_overview():
    header("소개 3", "프로젝트 개요 및 순서", "목표 → 평가 → 배출 세 관점으로 나누어 보고, 마지막에 등급과 배출 성과를 연결했습니다.")
    sb, kc, gir = load_sbti(), load_kcgs(), load_gir()
    near = share(sb["near_term_status"])["Targets set"]
    nz = share(sb["net_zero_status"])["Targets set"]
    ch = {p: share(kcgs_change(kc, *p)["등급변화"], ["상향", "유지", "하향"]) for p in PERIODS}
    bal = balanced_totals(gir)
    n_bal = int(gir.groupby("법인명")["대상년도"].nunique().eq(6).sum())
    drop = (bal.iloc[-1] / bal.iloc[0] - 1) * 100
    frames, _ = load_combined()
    _, df = active_frame(frames)
    rho_int, _, _ = safe_spearman(df, "환경등급순위", "배출집약도")
    rho_ghg, _, _ = safe_spearman(df, "환경등급순위", EMIS)

    steps = [
        ("1", "SBTi", "기업들은 감축목표를 어디까지 설정했나", f"단기 목표 설정 {near:.1f}%, Net-zero 목표 설정 {nz:.1f}%"),
        ("2", "KCGS", "국내 기업의 환경등급은 어떻게 변했나",
         f"2021→22년 하향 {ch[(2021, 2022)]['하향']:.1f}%, 2022→23년 상향 {ch[(2022, 2023)]['상향']:.1f}%"),
        ("3", "GIR", "국내 기업의 배출량은 어떻게 변했나", f"균형패널 {n_bal}개 기업 총배출량 {drop:.1f}%"),
        ("4", "등급과 배출 성과", "환경등급이 높은 기업은 배출 성과도 좋은가",
         f"등급과 배출집약도 ρ={rho_int:.2f}, 절대배출량 ρ={rho_ghg:+.2f}"),
    ]
    cols = st.columns(4)
    for col, (no, name, q, res) in zip(cols, steps):
        col.markdown(
            f'<div class="card"><span class="step-no">STEP {no}</span><h4>{name}</h4><p>{q}</p>'
            f'<div class="step-result">{res}</div></div>',
            unsafe_allow_html=True,
        )
    st.write("")
    st.markdown(
        '<div class="card"><span class="step-no">FINAL</span><h4>최종 분석</h4>'
        "<p>위 결과를 종합하고, 해석의 한계를 정리했다.</p></div>",
        unsafe_allow_html=True,
    )
    st.write("")
    st.subheader("사용 데이터")
    table(pd.DataFrame({
        "데이터": ["SBTi", "KCGS", "GIR", "DART"],
        "확인한 내용": ["기업의 감축목표 설정 상태", "환경등급", "온실가스 배출량", "매출액"],
        "기간": ["등록 현황 기준(연도 구분 없음)", "2020~2025", "2020~2025", "2020~2025"],
        "사용 단계": ["분석 1", "분석 2·4", "분석 3·4", "분석 4"],
    }))
    st.subheader("분석 원칙")
    callout(
        "• 표본이 작은 결과는 표본 수(n)를 함께 표시하고 일반화하지 않았다.<br>"
        "• 등급과 배출 성과의 관계는 상관관계로만 해석하고 인과관계로 단정하지 않았다.<br>"
        "• 결합 과정에서 중복과 충돌을 확인했고, 판단할 수 없는 값은 임의로 합치지 않고 제외했다."
    )
    st.caption("사용 도구: Python, Pandas, SciPy, Plotly, Streamlit")
    nav_footer(2)


# ──────────────────────────────────────────────────────────────
# 분석 1. SBTi
# ──────────────────────────────────────────────────────────────
def grouped_share_bar(cats, series: dict, colors: dict, title: str, key: str, height: int = 360):
    fig = go.Figure()
    for name, vals in series.items():
        fig.add_bar(name=name, x=cats, y=[vals[c] for c in cats], marker_color=colors[name],
                    text=[f"{vals[c]:.1f}%" for c in cats], textposition="outside",
                    hovertemplate=f"{name}<br>%{{x}}: %{{y:.1f}}%<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_yaxes(title_text="기업 비율(%)", range=[0, max(float(v.max()) for v in series.values()) * 1.18])
    style_fig(fig, title, height)
    plot(fig, key)


def stacked_status_bar(counts: pd.DataFrame, pcts: pd.DataFrame, title: str, key: str, height: int = 420):
    """가로 누적 막대. counts: 기업 수, pcts: 행 기준 비율(%)"""
    labels = [f"{i} (n={int(counts.loc[i].sum()):,})" for i in pcts.index]
    fig = go.Figure()
    for status in [s for s in STATUS_ORDER if s in pcts.columns]:
        vals = pcts[status]
        fig.add_bar(name=status, y=labels, x=vals, orientation="h", marker_color=STATUS_COLOR[status],
                    marker_line_color="white", marker_line_width=2,
                    text=[f"{v:.0f}%" if v >= 8 else "" for v in vals], textposition="inside",
                    textfont=dict(color="white"),
                    hovertemplate="%{y}<br>" + status + ": %{x:.1f}%<extra></extra>")
    fig.update_layout(barmode="stack")
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(title_text="비율(%)", range=[0, 100])
    style_fig(fig, title, height, horizontal=True)
    plot(fig, key)


def page_sbti():
    header("분석 1 · SBTi", "글로벌·한국 기업은 감축목표를 어디까지 설정했나",
           "SBTi에 등록된 기업의 단기(Near-term)·Net-zero 목표 설정 현황을 보고, 한국 기업을 전체와 비교했습니다.")
    df = load_sbti()
    kr = df[df["location"] == KOREA]
    near, nz = share(df["near_term_status"]), share(df["net_zero_status"])
    t = df[df["near_term_status"] == "Targets set"]["near_term_target_classification"]
    p15 = t.astype(str).str.contains("1.5", na=False).mean() * 100

    tabs = st.tabs(["전체 현황", "지역·산업", "한국과 전체 비교", "산업 구성 보정"])

    with tabs[0]:
        c = st.columns(4)
        kpi(c[0], "SBTi 등록 기업", f"{len(df):,}", "분석 대상 전체")
        kpi(c[1], "단기 목표 설정(Targets set)", pct(near["Targets set"]), "Near-term")
        kpi(c[2], "Scope 1·2 목표 중 1.5°C 수준", pct(p15), "목표 설정 기업 기준")
        kpi(c[3], "Net-zero 목표 설정", pct(nz["Targets set"]), f"단기 대비 {near['Targets set'] - nz['Targets set']:.1f}%p 낮음")
        st.write("")
        grouped_share_bar(STATUS_ORDER, {"Near-term": near, "Net-zero": nz},
                          {"Near-term": DEEP, "Net-zero": "#6DB593"}, "단기 목표 vs Net-zero 목표 상태", "sbti_status")
        callout("SBTi에 참여한 기업은 대부분 단기 감축목표 설정 단계까지 진행했지만, Net-zero 목표 설정까지 이어진 비중은 상대적으로 낮았습니다.")
        caveat("SBTi에 등록된 기업만을 기준으로 한 결과이며 전 세계 모든 기업의 현황이 아닙니다. 이 데이터만으로 실제 감축 진행 정도는 알 수 없습니다. "
               "Net-zero 상태가 비어 있는 기업 중에는 목표 문구에는 있으나 상태 컬럼에 반영되지 않은 예외도 있을 수 있습니다.")

    with tabs[1]:
        reg_n = df["region"].value_counts()
        reg = pd.crosstab(df["region"], df["near_term_status"])
        reg = reg.loc[reg_n.index]
        reg_pct = reg.div(reg.sum(axis=1), axis=0) * 100
        stacked_status_bar(reg, reg_pct, "지역별 단기 목표 상태 비율", "sbti_region", 380)
        top = df["sector"].value_counts().head(10).index
        sec = pd.crosstab(df[df["sector"].isin(top)]["sector"], df[df["sector"].isin(top)]["near_term_status"]).loc[top]
        sec_pct = sec.div(sec.sum(axis=1), axis=0) * 100
        stacked_status_bar(sec, sec_pct, "기업 수 상위 10개 산업별 단기 목표 상태 비율", "sbti_sector", 520)
        top_reg = reg_n.index[0]
        callout(f"기업 수는 {top_reg} 지역이 가장 많고({reg_n.iloc[0] / len(df) * 100:.1f}%), 산업별로도 목표 설정 단계에 차이가 있습니다.")
        caveat("지역·산업별 분포는 SBTi 등록 기업의 구성일 뿐, 특정 지역이 기후대응을 더 잘한다는 뜻이 아닙니다. 표본이 작은 지역은 비율 변동이 큽니다(막대 라벨의 n 참고).")

    with tabs[2]:
        n1, n2 = st.columns(2)
        with n1:
            grouped_share_bar(STATUS_ORDER, {"전체 SBTi": near, f"한국 (n={len(kr)})": share(kr["near_term_status"])},
                              {"전체 SBTi": C_GRAY, f"한국 (n={len(kr)})": C_GREEN}, "단기 목표 상태: 전체 vs 한국", "sbti_kr_near", 360)
        with n2:
            grouped_share_bar(STATUS_ORDER, {"전체 SBTi": nz, f"한국 (n={len(kr)})": share(kr["net_zero_status"])},
                              {"전체 SBTi": C_GRAY, f"한국 (n={len(kr)})": C_GREEN}, "Net-zero 목표 상태: 전체 vs 한국", "sbti_kr_nz", 360)
        kr_n = kr["sector"].value_counts()
        rows = []
        for s in kr_n[kr_n >= 5].index:
            a = kr[kr["sector"] == s]
            b = df[(df["sector"] == s) & (df["location"] != KOREA)]
            ka, kb = (a["near_term_status"] == "Targets set").mean() * 100, (b["near_term_status"] == "Targets set").mean() * 100
            rows.append({"산업": s, "한국 기업수": len(a), "한국 제외 기업수": len(b),
                         "한국 Targets set(%)": round(ka, 1), "한국 제외 Targets set(%)": round(kb, 1), "차이(%p)": round(ka - kb, 1)})
        tbl = pd.DataFrame(rows).sort_values("차이(%p)", ascending=False)
        fig = go.Figure(go.Bar(
            y=[f"{r['산업']} (한국 {r['한국 기업수']}·기타 {r['한국 제외 기업수']})" for _, r in tbl.iterrows()],
            x=tbl["차이(%p)"], orientation="h",
            marker_color=[C_GREEN if v >= 0 else C_AMBER for v in tbl["차이(%p)"]],
            text=[f"{v:+.1f}%p" for v in tbl["차이(%p)"]], textposition="outside",
            hovertemplate="%{y}<br>한국 − 한국 제외: %{x:+.1f}%p<extra></extra>"))
        fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
        fig.update_xaxes(title_text="한국 − 한국 제외 기업의 Targets set 비율 차이(%p)")
        style_fig(fig, "한국 기업 5개 이상 산업의 목표 설정 비율 차이", 60 + 44 * len(tbl), horizontal=True)
        plot(fig, "sbti_kr_sector")
        with st.expander("표로 보기"):
            table(tbl.set_index("산업"))
        caveat("한국 기업은 산업 세부로 갈수록 표본이 5~20개 수준입니다. 개별 산업 결과를 그 산업 전체의 특성으로 확대해서 해석하지 않았습니다.")

    with tabs[3]:
        w = kr["sector"].value_counts(normalize=True)
        rate = df[df["location"] != KOREA].groupby("sector")["near_term_status"].apply(lambda x: (x == "Targets set").mean())
        adj = pd.DataFrame({"w": w, "r": rate}).dropna()
        expected = (adj["w"] * adj["r"]).sum() * 100
        actual = (kr["near_term_status"] == "Targets set").mean() * 100
        c = st.columns(3)
        kpi(c[0], "한국 실제 Targets set", pct(actual), f"한국 기업 {len(kr)}개")
        kpi(c[1], "산업 구성 보정 기대 비율", pct(expected), "한국 산업 비중 × 한국 제외 산업별 비율")
        kpi(c[2], "차이", f"{actual - expected:+.1f}%p", f"보정 전 전체 대비 {actual - near['Targets set']:+.1f}%p")
        st.write("")
        top10 = kr["sector"].value_counts().head(10).index
        gs = df["sector"].value_counts(normalize=True) * 100
        ks = kr["sector"].value_counts(normalize=True) * 100
        fig = go.Figure()
        fig.add_bar(name="전체 SBTi", y=list(top10), x=[gs.get(s, 0) for s in top10], orientation="h", marker_color=C_GRAY,
                    text=[f"{gs.get(s, 0):.1f}%" for s in top10], textposition="outside")
        fig.add_bar(name="한국", y=list(top10), x=[ks[s] for s in top10], orientation="h", marker_color=C_GREEN,
                    text=[f"{ks[s]:.1f}%" for s in top10], textposition="outside")
        fig.update_layout(barmode="group", bargap=0.3)
        fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
        fig.update_xaxes(title_text="산업 비중(%)")
        style_fig(fig, "한국 상위 10개 산업의 비중: 한국 vs 전체", 560, horizontal=True)
        plot(fig, "sbti_mix")
        callout("한국과 전체의 목표 설정 비율 차이 중 일부는 산업 구성 차이와 관련되어 있을 가능성이 있지만, 산업 구성만으로 전체 차이를 설명하기는 어려웠습니다.")
        caveat("이 보정은 산업만 기준으로 한 것이며 기업 규모·지역·기업 유형 등 다른 요인은 반영하지 않았습니다.")
    nav_footer(3)


# ──────────────────────────────────────────────────────────────
# 분석 2. KCGS
# ──────────────────────────────────────────────────────────────
def page_kcgs():
    header("분석 2 · KCGS", "국내 기업의 환경등급은 어떻게 변했나",
           "2020~2025년 KCGS 환경등급의 분포와, 같은 기업의 등급 이동을 확인했습니다.")
    k = load_kcgs()
    c = st.columns(3)
    kpi(c[0], "분석 기간", "2020~2025", "2012~2019년은 미공개 등급이 많아 제외")
    kpi(c[1], "환경등급 기업연도", f"{len(k):,}", f"고유 기업 {k['기업코드'].nunique():,}개")
    kpi(c[2], "등급 체계", "A+ ~ D", "6단계")
    st.write("")
    tabs = st.tabs(["연도별 등급 분포", "동일 기업의 등급 이동", "이동표"])

    with tabs[0]:
        dist = pd.crosstab(k["평가년도"], k["환경"], normalize="index").mul(100).reindex(columns=GRADES)
        fig = go.Figure()
        for g in GRADES:
            fig.add_bar(name=g, x=[str(y) for y in dist.index], y=dist[g], marker_color=GRADE_COLOR[g],
                        marker_line_color="#8DB9A2" if g == "D" else "white", marker_line_width=1.5,
                        text=[f"{v:.0f}%" if v >= 6 else "" for v in dist[g]], textposition="inside",
                        textfont=dict(color="white" if g in ("A+", "A", "B+") else INK),
                        hovertemplate=f"{g}<br>%{{x}}년: %{{y:.1f}}%<extra></extra>")
        fig.update_layout(barmode="stack")
        fig.update_yaxes(title_text="기업 비율(%)", range=[0, 100])
        style_fig(fig, "연도별 환경등급 분포", 420)
        plot(fig, "kcgs_dist")
        with st.expander("표로 보기"):
            table(dist.round(1))
        callout("2022년에 D등급 비중이 크게 증가했고, 2023년부터 다시 감소했습니다. 원인은 다음 탭의 동일 기업 이동에서 확인합니다.")

    with tabs[1]:
        res = {p: kcgs_change(k, *p) for p in PERIODS}
        fig = go.Figure()
        xs = [f"{a}→{b}<br>n={len(res[(a, b)]):,}" for a, b in PERIODS]
        for name in ["상향", "유지", "하향"]:
            vals = [share(res[p]["등급변화"], ["상향", "유지", "하향"])[name] for p in PERIODS]
            fig.add_bar(name=name, x=xs, y=vals, marker_color=CHANGE_COLOR[name], marker_line_color="white", marker_line_width=2,
                        text=[f"{v:.1f}%" if v >= 4 else "" for v in vals], textposition="inside", textfont=dict(color="white"),
                        hovertemplate=f"{name}<br>%{{x}}: %{{y:.1f}}%<extra></extra>")
        fig.update_layout(barmode="stack")
        fig.update_yaxes(title_text="동일 기업 비율(%)", range=[0, 100])
        style_fig(fig, "같은 기업의 환경등급 이동 (상향·유지·하향)", 420)
        plot(fig, "kcgs_change")
        callout("2021→2022년에는 하향 기업이 많았고, 2022→2023년에는 상향이 크게 늘었습니다. 2023년 이후에는 등급을 유지하는 기업이 대부분입니다.")
        caveat("이 변화는 KCGS 평가등급의 변화입니다. 기업의 실제 환경성과가 같은 수준으로 변했다는 뜻은 아니며, 평가기준·모형 변화의 영향도 있을 수 있습니다.")

    with tabs[2]:
        label = st.selectbox("비교 기간", [f"{a}→{b}" for a, b in PERIODS], index=1, key="kcgs_period")
        y0, y1 = (int(x) for x in label.split("→"))
        m = kcgs_change(k, y0, y1)
        ct = pd.crosstab(m[f"환경_{y0}"], m[f"환경_{y1}"]).reindex(index=GRADES, columns=GRADES, fill_value=0)
        fig = px.imshow(ct, text_auto=True, aspect="auto", color_continuous_scale=["#F1F7F3", DEEP],
                        labels=dict(x=f"{y1}년 등급", y=f"{y0}년 등급", color="기업 수"))
        style_fig(fig, f"{y0}년 → {y1}년 환경등급 이동표 (기업 수)", 460)
        plot(fig, "kcgs_heat")
        low = ct.loc["D"]
        if low.sum() > 0:
            up = low.drop("D").sum()
            st.caption(f"{y0}년 D등급 기업 {int(low.sum())}개 중 {int(up)}개({up / low.sum() * 100:.1f}%)가 {y1}년에 상향되었습니다.")
    nav_footer(4)


# ──────────────────────────────────────────────────────────────
# 분석 3. GIR
# ──────────────────────────────────────────────────────────────
def balanced_totals(gir: pd.DataFrame) -> pd.Series:
    cnt = gir.groupby("법인명")["대상년도"].nunique()
    keep = cnt[cnt == 6].index
    return gir[gir["법인명"].isin(keep)].groupby("대상년도")[EMIS].sum()


def page_gir():
    header("분석 3 · GIR", "국내 기업의 온실가스 배출량은 어떻게 변했나",
           "기업-연도 단위 패널을 만들고, 2020~2025년 모두 관측되는 기업(균형패널)의 총배출량 추이를 확인했습니다.")
    g = load_gir()
    bal = balanced_totals(g)
    n_bal = g.groupby("법인명")["대상년도"].nunique().eq(6).sum()
    drop = (bal.iloc[-1] / bal.iloc[0] - 1) * 100
    c = st.columns(4)
    kpi(c[0], "기업-연도 관측치", f"{len(g):,}", "2020~2025 패널")
    kpi(c[1], "균형패널 기업", f"{n_bal:,}", "6개 연도 모두 관측")
    kpi(c[2], "2020 → 2025 총배출량", f"{drop:.2f}%", f"{bal.iloc[0] / 1e6:.1f}M → {bal.iloc[-1] / 1e6:.1f}M tCO₂eq")
    kpi(c[3], "배출량 결측", f"{int(g[EMIS].isna().sum())}건", "0으로 바꾸지 않고 보존")
    st.write("")
    a, b = st.columns([3, 2])
    with a:
        fig = go.Figure(go.Bar(x=[str(y) for y in bal.index], y=bal.values / 1e6, marker_color=C_GREEN,
                               text=[f"{v / 1e6:.1f}" for v in bal.values], textposition="outside",
                               hovertemplate="%{x}년: %{y:.1f}M tCO₂eq<extra></extra>"))
        fig.update_yaxes(title_text="총배출량(백만 tCO₂eq)", range=[0, bal.max() / 1e6 * 1.15])
        style_fig(fig, f"균형패널 {n_bal}개 기업의 총배출량", 380)
        plot(fig, "gir_total")
    with b:
        cnt = g["대상년도"].value_counts().sort_index()
        fig = go.Figure(go.Bar(x=[str(y) for y in cnt.index], y=cnt.values, marker_color=C_GRAY,
                               text=cnt.values, textposition="outside"))
        fig.update_yaxes(range=[0, cnt.max() * 1.15], title_text="기업 수")
        style_fig(fig, "연도별 관측 기업 수", 380)
        plot(fig, "gir_count")
    callout(f"균형패널 기준 총배출량은 2020년에서 2025년까지 {abs(drop):.2f}% 감소했습니다. 2021년에는 일시적으로 증가한 뒤 이후 지속적으로 감소했습니다.")
    with st.expander("기업-연도 패널 집계 원칙"):
        st.markdown(
            "- 업체 행이 1개이면 업체 값을 사용한다.\n"
            "- 업체 행이 여러 개이고 배출량이 같으면 한 번만 사용한다.\n"
            "- 업체 행이 여러 개이고 배출량이 다르면 자동 합산하지 않고 제외한다.\n"
            "- 업체 행이 없고 사업장이 1개이면 사업장 값을 사용한다.\n"
            "- 업체 행이 없고 실질적으로 서로 다른 사업장이 여러 개이면 합산한다.\n"
            "- 번호 또는 비고만 다른 실질 중복은 제거한다.\n"
            "- 2023년의 판단이 불가능한 근사 중복 3개 기업(에스케이피유코어, 에이치디씨폴리올, 쿠팡로지스틱스서비스)은 제외했다."
        )
    caveat("GIR은 온실가스 관리·할당 대상 기업 중심의 자료라 국내 전체 기업을 대표하지 않습니다. 총배출량 감소는 기업 구성이 고정된 균형패널 기준이며, 원인(감축 노력·생산 변화 등)은 이 분석으로 알 수 없습니다.")
    nav_footer(5)


# ──────────────────────────────────────────────────────────────
# 분석 4. 환경등급 × 배출 성과
# ──────────────────────────────────────────────────────────────
EXPORT_SNIPPET = '''# analysis4_reviewed.ipynb 맨 아래에 추가하는 셀
export_cols = ["기업키", "기업명", "평가년도", "성과연도", "환경", "환경등급순위",
               "온실가스 배출량(tCO₂eq)", "배출집약도", "지정업종(목표)/<br />계획업종(할당)"]
primary[export_cols].to_csv("../data/dashboard_primary.csv", index=False, encoding="utf-8-sig")
same_year[export_cols].to_csv("../data/dashboard_same_year.csv", index=False, encoding="utf-8-sig")'''


def page_combined():
    header("분석 4", "환경등급이 높은 기업은 실제 배출 성과도 좋은가",
           "KCGS 환경등급, GIR 온실가스 배출량, DART 매출액을 연결해 등급별 절대배출량과 배출집약도를 비교했습니다.")
    frames, is_temp = load_combined()
    if is_temp:
        st.warning("현재 **임시 데이터(동일연도 정렬)** 로 표시 중입니다. 노트북의 주 분석은 t-1 정렬이라 숫자가 다를 수 있습니다. "
                   "아래 '민감도' 탭의 안내대로 t-1 결합 파일을 내보내면 자동으로 주 분석 기준으로 바뀝니다.")
    keys = list(frames)
    basis = st.radio("정렬 기준", keys, horizontal=True, key="basis") if len(keys) > 1 else keys[0]
    df = frames[basis]
    if len(keys) > 1 and basis.startswith("t-1"):
        st.caption("평가년도 t의 환경등급을 t−1년의 배출량·매출액과 연결했습니다(KCGS 평가는 전년도 활동을 대상으로 하기 때문).")
    elif len(keys) > 1:
        st.caption("평가년도와 같은 해의 배출량·매출액을 연결했습니다(민감도 분석).")

    gs = (df.dropna(subset=["배출집약도"]).groupby("환경")
          .agg(n=("기업키", "size"), firms=("기업키", "nunique"), med=("배출집약도", "median")).reindex(GRADES))
    es = df.groupby("환경").agg(n=("기업키", "size"), med=(EMIS, "median")).reindex(GRADES)
    rho_i, _, n_i = safe_spearman(df, "환경등급순위", "배출집약도")
    rho_g, _, n_g = safe_spearman(df, "환경등급순위", EMIS)

    tabs = st.tabs(["등급별 배출 성과", "등급 변화와 집약도", "민감도 (t-1 vs 동일연도)", "기업별 보기"])

    with tabs[0]:
        c = st.columns(4)
        kpi(c[0], "분석 기업연도", f"{len(df):,}", f"고유 기업 {df['기업키'].nunique():,}개")
        kpi(c[1], "등급 vs 배출집약도", f"ρ = {rho_i:.2f}", "음(−): 등급 높을수록 집약도 낮은 경향")
        kpi(c[2], "등급 vs 절대배출량", f"ρ = {rho_g:+.2f}", "양(+): 등급 높을수록 배출량 큰 경향")
        kpi(c[3], "배출집약도 유효 관측치", f"{n_i:,}", "KRW 매출 > 0인 기업연도")
        st.write("")
        a, b = st.columns(2)
        with a:
            fig = go.Figure(go.Bar(
                x=GRADES, y=gs["med"], marker_color=[GRADE_COLOR[x] for x in GRADES],
                marker_line_color="#8DB9A2", marker_line_width=1,
                text=[f"{v:,.1f}" for v in gs["med"]], textposition="outside",
                customdata=np.stack([gs["n"], gs["firms"]], axis=-1),
                hovertemplate="%{x}등급<br>중앙값 %{y:,.1f}<br>기업연도 %{customdata[0]:,} · 기업 %{customdata[1]:,}<extra></extra>"))
            fig.update_yaxes(title_text="tCO₂eq / 매출 10억원", range=[0, np.nanmax(gs["med"]) * 1.18])
            fig.update_xaxes(title_text="환경등급")
            style_fig(fig, "등급별 배출집약도 (중앙값)", 400)
            plot(fig, "cmb_int")
        with b:
            fig = go.Figure(go.Bar(
                x=GRADES, y=es["med"], marker_color=[GRADE_COLOR[x] for x in GRADES],
                marker_line_color="#8DB9A2", marker_line_width=1,
                text=[f"{v:,.0f}" for v in es["med"]], textposition="outside",
                customdata=es["n"], hovertemplate="%{x}등급<br>중앙값 %{y:,.0f} tCO₂eq<br>기업연도 %{customdata:,}<extra></extra>"))
            fig.update_yaxes(title_text="tCO₂eq", range=[0, np.nanmax(es["med"]) * 1.18])
            fig.update_xaxes(title_text="환경등급")
            style_fig(fig, "등급별 절대배출량 (중앙값)", 400)
            plot(fig, "cmb_ghg")
        w_i, w_g = ("낮은" if rho_i < 0 else "높은"), ("큰" if rho_g > 0 else "작은")
        callout(f"이 정렬 기준에서는 등급이 높을수록 배출집약도는 {w_i} 경향(ρ={rho_i:.2f}), 절대배출량은 {w_g} 경향(ρ={rho_g:+.2f})이 "
                "나타났습니다. 절대배출량은 기업 규모의 영향이 크기 때문에 매출 대비 집약도를 함께 봐야 합니다.")
        with st.expander("분포까지 보기 (배출집약도, 로그 축)"):
            box = px.box(df.dropna(subset=["배출집약도"]), x="환경", y="배출집약도", log_y=True, points=False,
                         category_orders={"환경": GRADES}, color="환경", color_discrete_map=GRADE_COLOR)
            box.update_layout(showlegend=False)
            style_fig(box, "", 380)
            plot(box, "cmb_box")
        with st.expander("표로 보기"):
            out = gs.join(es["med"].rename("절대배출량 중앙값")).rename(
                columns={"n": "기업연도수", "firms": "고유기업수", "med": "집약도 중앙값"})
            table(out.round(1))
        caveat("Spearman 상관은 같은 기업이 여러 해 반복되는 패널이라 관측치가 독립적이지 않고, 상관관계는 인과관계가 아닙니다. 환경등급은 온실가스만으로 결정되지 않습니다.")

    with tabs[1]:
        ch = make_changes(df)
        opt = st.radio("비교 구간", ["최근 두 구간 (평가년도 2024·2025)", "전체 구간"], horizontal=True, key="chg_scope")
        use = ch[ch["평가년도"].isin([2024, 2025])] if opt.startswith("최근") else ch
        ct = change_table(use)
        rho_c, p_c, n_c = safe_spearman(use, "등급변화폭", "집약도증감률")
        c = st.columns(3)
        kpi(c[0], "유효 비교 건수", f"{n_c:,}", "등급 변화 + 집약도 변화 모두 계산 가능")
        kpi(c[1], "등급 변화폭 vs 집약도 변화율", f"ρ = {rho_c:.2f}", f"p = {p_c:.2g}")
        kpi(c[2], "하향 기업 집약도 변화(중앙값)", "–" if pd.isna(ct.loc["하향", "집약도증감률_중앙값"]) else f"{ct.loc['하향', '집약도증감률_중앙값']:+.1f}%", "")
        st.write("")
        vals = ct["집약도증감률_중앙값"]
        fig = go.Figure(go.Bar(
            x=[f"{k}<br>n={int(ct.loc[k, '유효비교수']):,}" for k in ct.index], y=vals,
            marker_color=[CHANGE_COLOR[k] for k in ct.index],
            text=[f"{v:+.1f}%" if pd.notna(v) else "" for v in vals], textposition="outside",
            hovertemplate="%{x}<br>집약도 증감률 중앙값 %{y:+.2f}%<extra></extra>"))
        fig.add_hline(y=0, line_color=MUTED, line_width=1)
        fig.update_yaxes(title_text="배출집약도 증감률 중앙값(%)")
        style_fig(fig, "등급 변화 방향별 배출집약도 증감률", 400)
        plot(fig, "cmb_change")
        fmt = lambda v: "–" if pd.isna(v) else f"{v:+.1f}%"
        up_v, keep_v, down_v = (ct.loc[k, "집약도증감률_중앙값"] for k in ["상향", "유지", "하향"])
        msg = f"상향·유지·하향 그룹의 집약도 증감률 중앙값은 각각 {fmt(up_v)}, {fmt(keep_v)}, {fmt(down_v)}입니다."
        if pd.isna(rho_c) or abs(rho_c) < 0.15 or p_c > 0.05:
            msg += f" 이 정렬·구간에서는 등급 변화와 집약도 변화의 관계가 뚜렷하지 않습니다(ρ={rho_c:.2f}, p={p_c:.2g})."
        elif rho_c < 0:
            msg += " 등급이 높아질수록 집약도가 줄어드는 방향의 관계가 나타났습니다."
        else:
            msg += " 등급이 높아질수록 집약도가 늘어나는 방향의 관계가 나타났습니다."
        callout(msg)
        with st.expander("분포까지 보기 (극단값은 삭제하지 않고 1~99% 구간만 표시)"):
            v = use.dropna(subset=["집약도증감률"])
            if len(v):
                lo, hi = v["집약도증감률"].quantile([0.01, 0.99])
                bx = px.box(v, x="등급변화구분", y="집약도증감률", points=False, color="등급변화구분",
                            category_orders={"등급변화구분": ["상향", "유지", "하향"]}, color_discrete_map=CHANGE_COLOR)
                bx.update_layout(showlegend=False)
                bx.update_yaxes(range=[lo, hi], title_text="집약도 증감률(%)")
                style_fig(bx, "", 380)
                plot(bx, "cmb_change_box")
        caveat("등급 변화는 평가기준 변화의 영향을 받을 수 있고, 관측된 관계는 인과관계가 아닙니다.")

    with tabs[2]:
        if len(frames) < 2:
            st.info("t-1 결합 파일이 아직 없어 비교할 수 없습니다. 노트북에서 아래 셀을 실행해 `data/` 폴더에 두 파일을 만들면 이 탭이 활성화됩니다.")
            st.code(EXPORT_SNIPPET, language="python")
        else:
            p_df, s_df = frames["t-1 (주 분석)"], frames["동일연도 (민감도)"]

            def recent_rho(frame):
                cc = make_changes(frame)
                cc = cc[cc["평가년도"].isin([2024, 2025])]
                return safe_spearman(cc, "등급변화폭", "집약도증감률")[0]

            rows = [
                {"항목": "등급 vs 절대배출량", "동일연도": safe_spearman(s_df, "환경등급순위", EMIS)[0],
                 "t-1": safe_spearman(p_df, "환경등급순위", EMIS)[0]},
                {"항목": "등급 vs 배출집약도", "동일연도": safe_spearman(s_df, "환경등급순위", "배출집약도")[0],
                 "t-1": safe_spearman(p_df, "환경등급순위", "배출집약도")[0]},
                {"항목": "최근 등급변화폭 vs 집약도변화율", "동일연도": recent_rho(s_df), "t-1": recent_rho(p_df)},
            ]
            sens = pd.DataFrame(rows).set_index("항목").round(4)
            table(sens)
            same_dir = all(np.sign(sens.iloc[i]["동일연도"]) == np.sign(sens.iloc[i]["t-1"]) for i in range(2))
            stronger = "t-1" if abs(sens.iloc[2]["t-1"]) > abs(sens.iloc[2]["동일연도"]) else "동일연도"
            callout(f"등급과 절대배출량·배출집약도의 관계는 두 정렬에서 방향이 {'같았고' if same_dir else '달랐고'}, "
                    f"최근 등급 변화와 집약도 변화의 관계는 {stronger} 정렬에서 더 뚜렷했습니다.")

    with tabs[3]:
        names = sorted(df["기업명"].dropna().unique())
        pick = st.selectbox("기업 선택", names, key="cmb_company")
        one = df[df["기업명"] == pick].sort_values("평가년도")
        a, b = st.columns(2)
        with a:
            fig = go.Figure(go.Scatter(x=one["평가년도"], y=one["환경등급순위"], mode="lines+markers",
                                       line=dict(color=DEEP, width=2, shape="hv"), marker=dict(size=9),
                                       customdata=one["환경"], hovertemplate="%{x}년 환경등급 %{customdata}<extra></extra>"))
            fig.update_yaxes(tickvals=[1, 2, 3, 4, 5, 6], ticktext=["D", "C", "B", "B+", "A", "A+"], range=[0.5, 6.5])
            fig.update_xaxes(dtick=1)
            style_fig(fig, "환경등급 추이", 320)
            plot(fig, "cmb_co_grade")
        with b:
            fig = go.Figure(go.Scatter(x=one["평가년도"], y=one["배출집약도"], mode="lines+markers",
                                       line=dict(color=C_GREEN, width=2), marker=dict(size=9),
                                       hovertemplate="%{x}년 %{y:,.1f}<extra></extra>"))
            fig.update_xaxes(dtick=1)
            fig.update_yaxes(title_text="tCO₂eq / 매출 10억원")
            style_fig(fig, "배출집약도 추이", 320)
            plot(fig, "cmb_co_int")
        table(one[["평가년도", "성과연도", "환경", EMIS, "배출집약도"]].round(1).reset_index(drop=True))
    nav_footer(6)


# ──────────────────────────────────────────────────────────────
# 최종 분석
# ──────────────────────────────────────────────────────────────
def page_final():
    header("최종 분석", "종합 결론", "각 분석에서 확인한 내용을 한곳에 모았습니다. 새로운 결론은 추가하지 않았습니다.")
    sb, kc, gir = load_sbti(), load_kcgs(), load_gir()
    near, nz = share(sb["near_term_status"])["Targets set"], share(sb["net_zero_status"])["Targets set"]
    bal = balanced_totals(gir)
    drop = (bal.iloc[-1] / bal.iloc[0] - 1) * 100
    frames, is_temp = load_combined()
    label, df = active_frame(frames)
    rho_i, _, _ = safe_spearman(df, "환경등급순위", "배출집약도")
    rho_g, _, _ = safe_spearman(df, "환경등급순위", EMIS)
    med = df.dropna(subset=["배출집약도"]).groupby("환경")["배출집약도"].median().reindex(GRADES)
    ch = make_changes(df)
    ch = ch[ch["평가년도"].isin([2024, 2025])]
    ct = change_table(ch)
    rho_c, _, n_c = safe_spearman(ch, "등급변화폭", "집약도증감률")

    c = st.columns(4)
    kpi(c[0], "SBTi 단기 목표 설정", pct(near), f"Net-zero는 {nz:.1f}%")
    kpi(c[1], "KCGS 2024→25 등급 유지", pct(share(kcgs_change(kc, 2024, 2025)["등급변화"], ["상향", "유지", "하향"])["유지"]), "2022~23년 큰 이동 이후 안정")
    kpi(c[2], "GIR 균형패널 총배출량", f"{drop:.1f}%", "2020 → 2025")
    kpi(c[3], "등급 vs 배출집약도", f"ρ = {rho_i:.2f}", label)
    st.write("")
    if is_temp:
        st.warning("결합 분석 수치는 임시 데이터(동일연도 정렬) 기준입니다. t-1 결합 파일을 내보내면 노트북의 주 분석 수치로 바뀝니다.")

    st.subheader("핵심 결과")
    def cell(k):
        v = ct.loc[k, "집약도증감률_중앙값"]
        return "–" if pd.isna(v) else f"{v:+.1f}%"
    st.markdown(
        f"1. **절대배출량**: 환경등급이 높을수록 배출량이 {'큰' if rho_g > 0 else '작은'} 경향이 나타났다(ρ={rho_g:+.2f}). 기업 규모 효과가 크다.\n"
        f"2. **배출집약도**: 등급이 높을수록 {'낮은' if rho_i < 0 else '높은'} 경향이 나타났다(ρ={rho_i:.2f}). 중앙값은 A+ {med['A+']:.1f}, D {med['D']:.1f} tCO₂eq/매출 10억원이다.\n"
        f"3. **등급 변화와 집약도**: 최근 동일 기업에서 등급 상향·유지·하향 그룹의 집약도 증감률 중앙값은 "
        f"각각 {cell('상향')}, {cell('유지')}, {cell('하향')}이었다(ρ={rho_c:.2f}, 유효비교 {n_c:,}건).\n"
        "4. 이 결과는 **상관관계**이며 인과관계가 아니다. KCGS 환경등급은 온실가스만으로 결정되지 않는다."
    )
    st.subheader("각 분석의 결과")
    st.markdown(
        f"- **SBTi**: 등록 기업의 {near:.1f}%가 단기 목표를 설정했지만 Net-zero 목표는 {nz:.1f}%였다.\n"
        "- **KCGS**: 2022~2023년에 등급 이동이 컸고 이후에는 유지 비중이 커졌다. 등급 변화는 실제 성과 변화와 같지 않다.\n"
        f"- **GIR**: 균형패널 기준 총배출량은 2020~2025년 {abs(drop):.1f}% 감소했다.\n"
        "- 위 세 분석은 각각 별도로 수행했고, 결합 분석은 KCGS·GIR·DART로 했다."
    )
    a, b = st.columns(2)
    with a:
        st.subheader("한계")
        st.markdown(
            "- GIR은 관리·할당 대상 기업 중심이다.\n"
            "- GIR 법인 경계와 DART 재무제표 경계가 완벽히 일치하지 않을 수 있다.\n"
            "- KCGS 환경등급은 온실가스만 평가하는 지표가 아니다.\n"
            "- 업종 분류체계가 연도별로 달라 업종 결과는 참고용으로만 썼다.\n"
            "- 같은 기업이 여러 연도 관측되어 단순 p-value의 독립성 가정에 제한이 있다.\n"
            "- 증감률의 극단값은 자동 삭제하지 않았다.\n"
            "- 상관관계는 인과관계를 의미하지 않는다."
        )
    with b:
        st.subheader("실무적 의미")
        callout("환경성과 관리는 ESG 환경등급 하나만 보는 것이 아니라 다음 항목을 함께 확인해야 한다.<br>"
                "• 절대배출량<br>• 매출 대비 배출집약도<br>• 전년 대비 변화<br>• 평가등급<br>• 감축목표")
    nav_footer(7)


# ──────────────────────────────────────────────────────────────
# 부록. 참고 자료 및 데이터
# ──────────────────────────────────────────────────────────────
DATA_DICT = {
    "sbti": [
        ("company_name", "기업명"),
        ("location", "소재 국가 (한국 기업은 'Korea, Republic of')"),
        ("region", "지역"),
        ("sector", "산업 (영문 원본 분류)"),
        ("organization_type", "기업 유형"),
        ("near_term_status", "단기(Near-term) 감축목표 상태: Targets set · Committed · Commitment removed (값이 없으면 미기재)"),
        ("near_term_target_classification", "단기 목표 수준 (예: 1.5°C)"),
        ("net_zero_status", "Net-zero 목표 상태 (값이 없으면 미기재)"),
    ],
    "kcgs": [
        ("기업명", "기업명"), ("기업코드", "종목코드 (6자리 문자열)"), ("ESG등급", "ESG 종합등급"),
        ("환경 / 사회 / 지배구조", "영역별 등급 (환경은 A+·A·B+·B·C·D)"), ("평가년도", "KCGS 평가 연도"),
    ],
    "gir": [
        ("관장기관", "배출량을 관리하는 주무 기관"), ("법인명", "기업(법인) 이름"), ("대상년도", "배출량 대상 연도"),
        ("지정구분", "업체 / 사업장 단위 구분"), (IND, "지정(목표) 또는 계획(할당) 업종"),
        (EMIS, "온실가스 배출량 (tCO₂eq)"), ("에너지 사용량(TJ)", "에너지 사용량 (TJ)"),
        ("검증수행기관", "배출량 검증 기관"), ("집계방식", "기업 단위로 합치는 방식 (업체 우선·단일 사업장 등)"),
    ],
    "combined": [
        ("기업키", "기업 식별자 (종목코드 기반, 없으면 정규화한 기업명)"), ("기업명", "기업명"),
        ("평가년도", "KCGS 환경등급 평가 연도 t"), ("성과연도", "연결한 배출량·매출액의 연도 (t-1 정렬이면 t−1)"),
        ("환경", "KCGS 환경등급"), ("환경등급순위", "D=1 … A+=6 (순서 표시용, 등급 간 거리가 같다는 뜻은 아님)"),
        (EMIS, "온실가스 배출량 (tCO₂eq)"), ("배출집약도", "배출량 ÷ 매출액(10억원), 단위 tCO₂eq/매출 10억원"),
        (IND, "GIR 원본 업종명"),
    ],
}

GIR_PREVIEW_COLS = ["관장기관", "법인명", "대상년도", "지정구분", IND, EMIS, "에너지 사용량(TJ)", "검증수행기관", "집계방식"]


def page_refs():
    header("부록", "참고 자료 및 데이터", "사용한 데이터의 범위, 전처리·결합 과정, 실제 데이터와 용어를 정리했습니다.")
    sb, kc, gir = load_sbti(), load_kcgs(), load_gir()
    frames, is_temp = load_combined()
    label, df = active_frame(frames)
    n_bal = int(gir.groupby("법인명")["대상년도"].nunique().eq(6).sum())
    tabs = st.tabs(["데이터 개요", "전처리·결합 흐름", "데이터 미리보기", "용어·다운로드"])

    with tabs[0]:
        c = st.columns(4)
        kpi(c[0], "SBTi 등록 기업", f"{len(sb):,}", "한국 기업 125개")
        kpi(c[1], "KCGS 환경등급", f"{len(kc):,}", f"2020~2025 기업연도 · 기업 {kc['기업코드'].nunique():,}개")
        kpi(c[2], "GIR 기업-연도", f"{len(gir):,}", f"균형패널 {n_bal}개 기업")
        kpi(c[3], "결합 데이터", f"{len(df):,}", f"기업연도 · {label}")
        st.write("")
        table(pd.DataFrame({
            "데이터": ["SBTi", "KCGS", "GIR", "DART", "결합 데이터"],
            "파일": ["sbti.csv", "kcgs.csv", "gir_company_2020_2025_reviewed.csv (원본 gir_2020~2025.csv)",
                     "dart_pl_YYYY.csv (Git LFS)",
                     "dashboard_primary.csv" if not is_temp else "esg_environment_final.csv (임시)"],
            "확인한 내용": ["기업의 감축목표 설정 상태", "ESG·환경 등급", "기업 온실가스 배출량·에너지 사용량", "매출액(ifrs-full_Revenue)",
                        "환경등급 + 배출량 + 매출액 + 배출집약도"],
            "기간": ["등록 현황 기준", "2020~2025", "2020~2025", "2020~2025", "2020~2025"],
            "사용 분석": ["분석 1", "분석 2·4", "분석 3·4", "분석 4 (앱에는 미포함)", "분석 4·최종"],
        }))
        caveat("DART 원본은 용량이 커서 Git LFS로 관리되며, 이 앱은 DART 매출액이 이미 결합된 결과 파일만 사용합니다.")

    with tabs[1]:
        callout("<b>연결 구조</b><br>SBTi는 별도로 분석했고(분석 1), KCGS와 GIR은 각각 정리한 뒤(분석 2·3) "
                "<b>KCGS + GIR + DART</b>를 결합해 등급과 배출 성과를 비교했습니다(분석 4).")
        s1, s2 = st.columns(2)
        with s1:
            st.markdown("#### SBTi")
            st.markdown(
                "- 값이 전부 비어 있는 `Unnamed` 컬럼 4개를 삭제했다.\n"
                "- 목표연도 컬럼은 값과 실제 목표 문구가 일치하지 않는 경우가 있어 분석에서 **제외**했다.\n"
                "- 한국 기업은 `location == 'Korea, Republic of'`로 추출했다(125개).\n"
                "- Net-zero 상태가 비어 있으면 `미기재`로 표시하고, 원본 상태값은 그대로 사용했다."
            )
            st.markdown("#### KCGS")
            st.markdown(
                "- 2020~2025년, 실제 등급(A+~D)이 있는 환경등급만 사용했다(2012~2019년은 미공개가 많아 제외).\n"
                "- 등급 이동 분석은 **기업코드**로 같은 기업을 연결했다.\n"
                "- 상향·하향 판단을 위해 등급을 D=1 … A+=6 순위로 바꿨다(등급 간 거리가 같다는 뜻은 아님)."
            )
            st.markdown("#### GIR")
            st.markdown(
                "- 원본 6개 파일(2020~2025)로 기업-연도 패널을 다시 만들었다.\n"
                "- 업체 행이 1개면 그 값, 여러 개이고 배출량이 같으면 한 번만, 다르면 자동 합산하지 않고 제외했다.\n"
                "- 업체 행이 없으면 단일 사업장 값을 쓰고, 서로 다른 사업장이 여러 개면 합산했다.\n"
                f"- 결과: **{len(gir):,}행**, 기업명+연도 중복 0개, 음수 0개, 배출량 결측 {int(gir[EMIS].isna().sum())}개(0으로 바꾸지 않고 보존).\n"
                f"- 6개 연도 모두 있는 균형패널은 **{n_bal}개 기업**이다."
            )
        with s2:
            st.markdown("#### DART")
            st.markdown(
                "- 매출액은 `ifrs-full_Revenue` 항목만 사용했다.\n"
                "- 종목코드는 정확히 숫자 6자리인 것만 허용하고, 영문이 섞인 코드는 강제로 바꾸지 않았다.\n"
                "- 기업·연도·별도/연결별로 최신 결산일을 선택하고, 같은 결산일에 서로 다른 매출이 있으면 분석을 중단하도록 했다(충돌 0건).\n"
                "- 배출 경계가 법인 단위인 GIR에 맞추기 위해 **별도재무제표를 우선**했다."
            )
            st.markdown("#### 결합 (분석 4)")
            st.markdown(
                "- 기업명에서 `주식회사·(주)·㈜·유한회사·(유)`와 공백을 제거해 정규화한 이름 + 연도로 KCGS와 GIR을 연결했다"
                "(정규화 후 중복·충돌 0건).\n"
                "- KCGS 종목코드와 성과연도로 DART 매출액을 붙였다.\n"
                "- **주 분석(t-1)**: 평가년도 t ↔ 배출량·매출액 t−1 → 1,149 기업연도 · 270개 기업. "
                "**민감도**: 동일연도 정렬.\n"
                "- **배출집약도** = 배출량 ÷ (매출액 ÷ 10억원). 통화가 KRW이고 매출액이 0보다 큰 경우만 계산했다(t-1 기준 1,141 기업연도).\n"
                "- 위 결합 수치는 노트북(`analysis4_reviewed`) t-1 정렬 기준이다."
            )
        caveat("GIR 법인 경계와 DART 재무제표 경계가 완전히 일치하지는 않으며, 별도재무제표 우선 선택도 이 차이를 완전히 없애지는 못합니다.")

    with tabs[2]:
        prev = {
            "SBTi · 기업 목표 현황": (sb, None, "sbti"),
            "KCGS · 환경등급": (kc[["기업명", "기업코드", "ESG등급", "환경", "사회", "지배구조", "평가년도"]], "평가년도", "kcgs"),
            "GIR · 기업-연도 패널": (gir[GIR_PREVIEW_COLS], "대상년도", "gir"),
            f"결합 데이터 · {label}": (df, "평가년도", "combined"),
        }
        name = st.selectbox("데이터 선택", list(prev), key="prev_ds")
        data, ycol, dkey = prev[name]
        if ycol:
            years = ["전체"] + [int(v) for v in sorted(data[ycol].dropna().unique())]
            year = st.selectbox("연도", years, key="prev_year")
            if year != "전체":
                data = data[data[ycol] == year]
        c = st.columns(3)
        kpi(c[0], "행 수", f"{len(data):,}", "")
        kpi(c[1], "컬럼 수", f"{data.shape[1]:,}", "")
        kpi(c[2], "결측 셀", f"{int(data.isna().sum().sum()):,}", "")
        st.write("")
        table(data.head(1000).reset_index(drop=True), height=420)
        if len(data) > 1000:
            st.caption(f"전체 {len(data):,}행 중 상위 1,000행을 표시합니다. 아래 버튼으로 선택한 전체 데이터를 내려받을 수 있습니다.")
        st.markdown("**컬럼 설명**")
        table(pd.DataFrame(DATA_DICT[dkey], columns=["컬럼", "설명"]).set_index("컬럼"))
        st.download_button("선택한 데이터 내려받기 (CSV)", data.to_csv(index=False).encode("utf-8-sig"),
                           file_name="preview_data.csv", mime="text/csv", key="prev_dl")

    with tabs[3]:
        st.subheader("용어")
        with st.expander("SBTi · KCGS · GIR · DART · 배출집약도 · Spearman ρ", expanded=True):
            st.markdown(
                "- **SBTi**: 기업의 온실가스 감축목표가 과학적 기준에 맞는지 검증하는 이니셔티브. Near-term은 단기, Net-zero는 넷제로 목표.\n"
                "- **KCGS**: 국내 기업 ESG 평가 기관. 환경 등급은 A+ ~ D.\n"
                "- **GIR**: 국내 기업의 온실가스 배출량·에너지 사용량 자료(관리·할당 대상 기업 중심).\n"
                "- **DART**: 전자공시 재무제표. 이 프로젝트에서는 매출액만 사용.\n"
                "- **배출집약도**: 온실가스 배출량 ÷ 매출액(10억원). 기업 규모의 영향을 줄이기 위한 지표.\n"
                "- **Spearman ρ**: 순위 기반 상관계수. −1~+1이며 인과관계를 뜻하지 않음."
            )
        st.subheader("다운로드")
        c1, c2 = st.columns(2)
        c1.download_button("GIR 기업-연도 패널 (CSV)", gir.to_csv(index=False).encode("utf-8-sig"),
                           file_name="gir_panel.csv", mime="text/csv", key="dl_gir")
        c2.download_button(f"결합 데이터 · {label} (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                           file_name="combined.csv", mime="text/csv", key="dl_combined")
        st.caption("사용 도구: Python, Pandas, SciPy, Plotly, Streamlit")
    nav_footer(8)


# ──────────────────────────────────────────────────────────────
# 내비게이션
# ──────────────────────────────────────────────────────────────
P = st.Page
PAGE_LIST = [
    P(page_esg, title="ESG란", icon=":material/eco:", default=True),
    P(page_why, title="이 프로젝트를 하는 이유", icon=":material/help:", url_path="why"),
    P(page_overview, title="프로젝트 개요 및 순서", icon=":material/route:", url_path="overview"),
    P(page_sbti, title="분석 1 · SBTi 감축목표", icon=":material/public:", url_path="sbti"),
    P(page_kcgs, title="분석 2 · KCGS 환경등급", icon=":material/grade:", url_path="kcgs"),
    P(page_gir, title="분석 3 · GIR 배출량", icon=":material/factory:", url_path="gir"),
    P(page_combined, title="분석 4 · 등급과 배출 성과", icon=":material/insights:", url_path="combined"),
    P(page_final, title="최종 분석", icon=":material/flag:", url_path="final"),
    P(page_refs, title="참고 자료 · 데이터", icon=":material/folder_open:", url_path="refs"),
]
ORDER = PAGE_LIST
NAV = {"소개": PAGE_LIST[0:3], "분석": PAGE_LIST[3:8], "부록": PAGE_LIST[8:9]}

st.sidebar.caption(f"{PROJECT_TITLE}\n\n{PROJECT_SUBTITLE}")
pg = st.navigation(NAV)
pg.run()
