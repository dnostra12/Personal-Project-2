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
    page_icon="🌳",
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

# 범주 색: 딥그린 · 민트 · 더스티 로즈(부정 방향) · 연회색(중립)
C_GREEN = "#1B7F5C"
C_AMBER = "#B65C74"     # 더스티 로즈 (하향·철회 등 부정 방향)
C_BLUE = "#86CBA5"
C_GRAY = "#D5DDD8"     # 중립(미기재·기준 집단)
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
STATUS_KO = {"Targets set": "목표 설정", "Committed": "설정 약속", "Commitment removed": "약속 철회", "미기재": "미기재"}
TEXT_ON = {"Targets set": "white", "Committed": INK, "Commitment removed": "white", "미기재": INK}
REGION_KO = {
    "Europe": "유럽", "Asia": "아시아", "Northern America": "북미", "Latin America and the Caribbean": "중남미",
    "Oceania": "오세아니아", "Africa": "아프리카", "MENA": "중동·북아프리카",
}
SECTOR_KO = {
    "Professional Services": "전문서비스",
    "Electrical Equipment and Machinery": "전기장비·기계",
    "Software and Services": "소프트웨어·IT서비스",
    "Food and Beverage Processing": "식음료",
    "Construction and Engineering": "건설·엔지니어링",
    "Textiles, Apparel, Footwear and Luxury Goods": "섬유·의류·럭셔리",
    "Trading Companies and Distributors, and Commercial Services and Supplies": "유통·상사업",
    "Consumer Durables, Household and Personal Products": "내구소비재·생활용품",
    "Pharmaceuticals, Biotechnology and Life Sciences": "제약·바이오",
    "Automobiles and Components": "자동차·부품",
    "Technology Hardware and Equipment": "기술 하드웨어",
    "Containers and Packaging": "용기·포장",
    "Chemicals": "화학",
    "Real Estate": "부동산",
    "Retailing": "소매",
    "Banks, Diverse Financials, Insurance": "금융",
    "Building Products": "건축자재",
    "Ground Transportation - Trucking Transportation": "육상운송(트럭)",
    "Telecommunication Services": "통신",
    "Forest and Paper Products - Forestry, Timber, Pulp and Paper, Rubber": "산림·제지·고무",
    "Healthcare Providers and Services, and Healthcare Technology": "의료서비스·헬스테크",
    "Media": "미디어",
    "Hotels, Restaurants and Leisure, and Tourism Services": "호텔·외식·레저",
    "Food Production - Agricultural Production": "농업 생산",
    "Food and Staples Retailing": "식료품 소매",
    "Electric Utilities and Independent Power Producers and Energy Traders (including Fossil, Alternative and Nuclear Energy)": "전력·발전",
    "Construction Materials": "건설자재",
    "Semiconductors and Semiconductors Equipment": "반도체",
    "Solid Waste Management Utilities": "폐기물 처리",
    "Healthcare Equipment and Supplies": "의료기기",
    "Mining - Iron, Aluminum, Other Metals": "광업(철·알루미늄 등)",
    "Specialized Consumer Services": "전문 소비자서비스",
    "Air Freight Transportation and Logistics": "항공화물·물류",
    "Food Production - Animal Source Food Production": "축산 식품",
    "Ground Transportation - Railroads Transportation": "철도 운송",
    "Homebuilding": "주택건설",
    "Specialized Financial Services, Consumer Finance, Insurance Brokerage Firms": "특수금융·소비자금융",
    "Aerospace and Defense": "항공우주·방위",
    "Water Utilities": "수도",
    "Ground Transportation - Highways and Railtracks": "도로·철도 인프라",
    "Education Services": "교육 서비스",
    "Air Transportation - Airlines": "항공사",
    "Water Transportation - Water Transportation": "해운",
    "Mining - Other (Rare Minerals, Precious Metals and Gems)": "광업(희소·귀금속)",
    "Tires": "타이어",
    "Air Transportation - Airport Services": "공항 서비스",
    "Water Transportation - Ports and Services": "항만·서비스",
    "Tobacco": "담배",
    "Public Agencies": "공공기관",
    "Gas Utilities": "가스",
    "Mining - Coal": "석탄 광업",
}


def ko_region(name: str) -> str:
    return REGION_KO.get(name, name)


def ko_sector(name: str) -> str:
    return SECTOR_KO.get(name, name)


SVG_PATHS = {
    "sprout": '<path d="M24 43V25"/><path d="M24 27c0-8-5.5-12-13-12 0 8 5.5 12 13 12z"/>'
              '<path d="M24 22c0-6.5 4.5-10.5 12-10.5 0 6.5-4.5 10.5-12 10.5z"/>',
    "tree": '<circle cx="24" cy="17" r="11"/><path d="M24 28v15"/><path d="M24 36l-6-5"/><path d="M24 33l5-4"/>',
    "forest": '<path d="M24 44v-6"/><path d="M24 38H13l11-15 11 15z"/><path d="M24 26H17l7-9 7 9z"/>'
              '<path d="M9 44v-4"/><path d="M9 40H3l6-8 6 8z"/><path d="M39 44v-4"/><path d="M39 40h-6l6-8 6 8z"/>',
}


def svg_icon(kind: str, size: int = 40, color: str = C_GREEN, stroke: float = 2.4) -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 48 48" fill="none" stroke="{color}" '
            f'stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" '
            f'xmlns="http://www.w3.org/2000/svg">{SVG_PATHS[kind]}</svg>')
FONT = "Pretendard, 'Noto Sans KR', 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif"

CSS = f"""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css');
html, body {{ font-family: {FONT}; }}
.stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp td, .stApp th, .stApp button, .stApp input, .stApp textarea {{ font-family: {FONT}; }}
.stApp {{ background: #F7FAF8; color: {INK}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 4.6rem; padding-bottom: 3rem; max-width: 1180px; }}
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
.hero {{ background: #FFFFFF; border: 1px solid #DCE8E1; border-left: 6px solid {C_GREEN}; border-radius: 14px;
         padding: 26px 30px; margin-bottom: 1.4rem; }}
.hero .eyebrow {{ color: {C_GREEN}; font-size: .8rem; font-weight: 700; letter-spacing: .08em; }}
.hero .title {{ color: {DEEP}; font-size: 2.1rem; font-weight: 800; line-height: 1.25; margin: .25rem 0 .2rem; }}
.hero .subtitle {{ color: {SUB}; font-size: 1.1rem; font-weight: 500; }}
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
.caveat {{ border-left: 4px solid #5E86AD; background: #EEF3F8; padding: 12px 16px;
           border-radius: 0 10px 10px 0; color: {INK}; margin: .6rem 0 1rem; line-height: 1.65; }}
.body-text {{ font-size: 1.02rem; line-height: 1.85; color: {INK}; }}
.quote {{ font-size: 1.25rem; font-weight: 700; color: {DEEP}; border-left: 5px solid {C_GREEN};
          padding: 6px 0 6px 16px; margin: 1.1rem 0; }}
.card .icon {{ margin-bottom: 10px; line-height: 0; }}
.chart-title {{ font-weight: 700; font-size: 1.08rem; color: {DEEP}; margin: 2px 0 0; }}
.chart-sub {{ color: {SUB}; font-size: .88rem; margin: 3px 0 6px; line-height: 1.5; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 14px; background: #FFFFFF; }}
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
    """제목·해설은 차트 밖(카드 상단)에 두고, 차트는 카드 안에 넣는다."""
    meta = fig.layout.meta
    meta = meta if isinstance(meta, dict) else {}
    title, sub = meta.get("title", ""), meta.get("sub", "")
    with st.container(border=True):
        if title:
            head = f'<div class="chart-title">{title}</div>'
            if sub:
                head += f'<div class="chart-sub">{sub}</div>'
            st.markdown(head, unsafe_allow_html=True)
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


def style_fig(fig: go.Figure, title: str = "", height: int = 380, horizontal: bool = False,
              sub: str = "", hide_y: bool = False) -> go.Figure:
    """공통 차트 스타일: 큰 글씨, 옅은 격자, 둥근 막대, 겹치는 라벨은 자동 숨김."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        meta={"title": title, "sub": sub},
        font=dict(family=FONT, color=INK, size=14),
        margin=dict(l=8, r=28 if horizontal else 12, t=44 if len(fig.data) > 1 else 12, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title_text="", font=dict(size=13)),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(size=13, color=INK)),
        uniformtext=dict(minsize=11, mode="hide"),
    )
    if fig.layout.bargap is None:
        fig.update_layout(bargap=0.35)
    tick_sub, tick_main = dict(size=13, color=SUB), dict(size=13, color=INK)
    if horizontal:  # 가로 막대: 값 축(x)에만 옅은 격자
        fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False, showline=False, tickfont=tick_sub)
        fig.update_yaxes(showgrid=False, showline=False, tickfont=tick_main)
    else:
        fig.update_xaxes(showgrid=False, showline=True, linecolor=GRID, tickfont=tick_main)
        fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, showline=False, tickfont=tick_sub)
    if hide_y:  # 막대 위에 값을 직접 적은 차트는 y축을 숨겨 깔끔하게
        fig.update_yaxes(visible=False)
    try:
        fig.update_traces(cliponaxis=False, textfont_size=13, selector=dict(type="bar"))
        if fig.layout.barmode != "stack":  # 누적 막대는 조각마다 둥글면 어색해서 제외
            fig.update_traces(marker_cornerradius=6, selector=dict(type="bar"))
    except Exception:  # 오래된 plotly에서 일부 속성이 없을 때도 앱은 그대로 동작
        pass
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
def headline_kpis() -> None:
    """첫 화면 맨 위: 결론을 먼저 보여주는 핵심 결과 4개."""
    sb, kc, gir = load_sbti(), load_kcgs(), load_gir()
    near = share(sb["near_term_status"])["Targets set"]
    bal = balanced_totals(gir)
    drop = (bal.iloc[-1] / bal.iloc[0] - 1) * 100
    frames, is_temp = load_combined()
    label, df = active_frame(frames)
    rho_i, _, _ = safe_spearman(df, "환경등급순위", "배출집약도")
    med = df.dropna(subset=["배출집약도"]).groupby("환경")["배출집약도"].median().reindex(GRADES)
    st.markdown('<div class="kicker">핵심 결과</div>', unsafe_allow_html=True)
    c = st.columns(4)
    kpi(c[0], "SBTi 단기 목표 설정", pct(near), "등록 기업 중 목표를 설정한 비율")
    kpi(c[1], "GIR 총배출량 (2020→2025)", f"{drop:.1f}%", "6개 연도 모두 관측된 기업 기준")
    kpi(c[2], "등급 vs 배출집약도", f"ρ = {rho_i:.2f}", "등급 높을수록 집약도 낮은 경향" if rho_i < 0 else "등급 높을수록 집약도 높은 경향")
    kpi(c[3], "집약도 중앙값 (A+ vs D)", f"{med['A+']:.0f} vs {med['D']:.0f}", "tCO₂eq / 매출 10억원")
    st.caption("등급과 배출 성과의 관계는 상관관계이며 인과관계가 아닙니다." + (" 현재 임시(동일연도) 데이터 기준입니다." if is_temp else f" 기준: {label}."))
    st.write("")


def page_esg():
    st.markdown(
        f'<div class="hero"><div class="eyebrow">PERSONAL PROJECT 2</div>'
        f'<div class="title">{PROJECT_TITLE}</div><div class="subtitle">{PROJECT_SUBTITLE}</div></div>',
        unsafe_allow_html=True,
    )
    headline_kpis()
    header("소개 1", "ESG란", "기업을 재무성과만이 아니라 환경·사회·지배구조까지 함께 보는 관점입니다.", banner=False)
    body(
        "ESG는 기업을 평가할 때 재무성과뿐 아니라 환경(Environment), 사회(Social), 지배구조(Governance) "
        "측면의 지속가능성을 함께 보는 개념이다."
    )
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="card"><div class="icon">{svg_icon("sprout")}</div><h4>환경 (E)</h4><p>온실가스 배출, 에너지 사용, 폐기물·수질 관리처럼 '
                "기업 활동이 환경에 미치는 영향을 다룬다.</p></div>", unsafe_allow_html=True)
    c2.markdown(f'<div class="card"><div class="icon">{svg_icon("forest")}</div><h4>사회 (S)</h4><p>노동·안전, 인권, 지역사회처럼 '
                "사람과 관련된 기업의 책임을 다룬다.</p></div>", unsafe_allow_html=True)
    c3.markdown(f'<div class="card"><div class="icon">{svg_icon("tree")}</div><h4>지배구조 (G)</h4><p>이사회 구성, 경영 투명성, 주주 권리처럼 '
                "기업이 운영되는 방식을 다룬다.</p></div>", unsafe_allow_html=True)
    st.write("")
    body("이 프로젝트에서는 그중 <b>환경(E), 특히 온실가스</b>에 집중했다. 기업의 환경 대응을 아래 네 가지 자료로 나누어 살펴봤다.")
    st.write("")
    t_sbti, t_kcgs, t_gir, t_dart = st.tabs(["SBTi", "KCGS 환경등급", "GIR", "DART · 배출집약도"])
    with t_sbti:
        st.markdown("##### SBTi (Science Based Targets initiative)")
        body("기업이 세운 온실가스 감축목표가 <b>파리협정 수준의 과학적 기준</b>에 맞는지 검증해 주는 국제 이니셔티브다. "
             "2015년 CDP, 유엔글로벌콤팩트(UNGC), 세계자원연구소(WRI), 세계자연기금(WWF) 등이 함께 만들었다.")
        st.markdown(
            "- **단기 목표(Near-term)**: 보통 5~10년 안에 달성할 감축목표\n"
            "- **Net-zero 목표**: 2050년경까지 배출을 거의 0에 가깝게 줄이는 장기 목표\n"
            "- **기업 상태**: 목표 설정(Targets set) · 설정 약속(Committed) · 약속 철회(Commitment removed)"
        )
        caveat("SBTi는 <b>목표가 과학적으로 타당한지</b>를 보는 제도이지, 실제로 배출을 줄였는지를 보여주는 자료는 아니다. "
               "그래서 이 프로젝트에서는 '목표 수준'을 보는 자료로 쓰고, 실제 배출 성과는 GIR로 따로 확인했다.")
    with t_kcgs:
        st.markdown("##### KCGS 환경등급")
        body("한국ESG기준원(KCGS)은 코스피 상장사 전체와 코스닥의 영향이 큰 기업 등을 대상으로 매년 ESG 평가를 한다"
             "(2024년 1,068개사). 환경(E)·사회(S)·지배구조(G) 영역별 등급과 통합등급을 부여하며, "
             "이 프로젝트는 그중 <b>환경(E) 영역 등급</b>만 사용했다.")
        st.markdown("**등급 체계 — 7단계 절대평가**")
        st.markdown("`S (탁월)` › `A+ (매우 우수)` › `A (우수)` › `B+ (양호)` › `B (보통)` › `C (취약)` › `D (매우 취약)`")
        st.caption("다른 기업과의 순위가 아니라 점수 기준에 따라 등급이 정해지는 절대평가다. "
                   "이 프로젝트의 데이터에는 S등급 기업이 없고, '미공개'·'등급없음'은 분석에서 제외했다.")
        st.markdown("**등급은 이렇게 산정된다**")
        c1, c2, c3 = st.columns(3)
        c1.markdown('<div class="card"><h4>① 기본평가 (득점)</h4><p>환경경영 체계·조직·활동처럼 '
                    "바람직한 관행을 갖췄는지 문항별로 득점한다. (예: 조직 체계가 있으면 득점, 없으면 0점)</p></div>", unsafe_allow_html=True)
        c2.markdown('<div class="card"><h4>② 심화평가 (감점)</h4><p>법적·행정적 제재나 환경 사고 같은 '
                    "중대 이슈가 있으면 정해진 감점 한도 안에서 점수를 깎는다.</p></div>", unsafe_allow_html=True)
        c3.markdown('<div class="card"><h4>③ 등급 부여</h4><p>기본 득점에서 감점을 뺀 점수로 영역별 등급을 산출하고, '
                    "ESG기준위원회 심의를 거쳐 최종 등급이 정해진다. 정기등급은 매년 10월에 발표한다.</p></div>", unsafe_allow_html=True)
        st.write("")
        st.markdown("**환경(E) 평가지표 — 대분류 4개, 중분류 12개**")
        st.markdown(
            "- **리더십과 거버넌스**: 거버넌스, 전략 및 목표, 환경경영 내재화\n"
            "- **위험관리**: 환경 위험관리 체계\n"
            "- **운영 및 성과**: 기후변화, 자원순환, 물·토양·생물다양성, 오염물질·화학물질, 친환경 공급망, 친환경 제품·서비스\n"
            "- **이해관계자 소통**: 이해관계자 대응, 환경정보 공개"
        )
        st.caption("평가 대상은 환경 영향도를 고려해 21개 산업(철강/비철, 석유/화학, 전기/전자, 건설, 운수, 금융 등)으로 나누고, "
                   "산업에 맞는 지표를 적용한다.")
        callout("<b>이 프로젝트에서 중요한 점</b>: 환경등급은 실제 배출량을 직접 재는 지표가 아니라, "
                "<b>환경경영 체계와 공시 수준을 평가</b>한 결과다. 그래서 '등급이 높은 기업이 실제로 배출도 적은가'를 "
                "GIR 배출량과 대조하는 것이 이 분석의 핵심 질문이 된다.")
        st.caption("출처: 한국ESG기준원, 「ESG 평가 방법론」(2026.2). 등급별 점수 구간과 문항별 배점은 이 자료에 공개되어 있지 않아 적지 않았다.")
    with t_gir:
        st.markdown("##### GIR (온실가스종합정보센터)")
        body("국가 온실가스 통계와 정보를 관리하는 정부 기관이다. 온실가스 <b>목표관리제 관리업체</b>와 "
             "<b>배출권거래제 할당대상업체</b>가 제출해 검증을 마친 <b>명세서 배출량</b>(연도별 tCO₂eq)과 에너지 사용량을 공개한다.")
        st.markdown(
            "- **장점**: 검증을 거친 공식 배출량이라 신뢰도가 높다.\n"
            "- **한계**: 규제 대상 기업만 포함되어 있어, 모든 상장사를 다루지는 못한다.\n"
            "- **이 프로젝트의 처리**: 연도별 명세서를 기업 단위로 정리하고, 전 기간 관측되는 기업으로 균형패널을 만들어 총배출량 추이를 계산"
        )
    with t_dart:
        st.markdown("##### DART · 배출집약도")
        body("배출량은 기업 규모가 클수록 커지기 때문에 그대로 비교하면 규모 차이가 결과를 좌우한다. "
             "그래서 금융감독원 전자공시(DART)의 <b>매출액</b>으로 나눈 <b>배출집약도</b>를 함께 썼다.")
        st.markdown("**배출집약도 = 온실가스 배출량(tCO₂eq) ÷ 매출액(10억원)**")
    st.write("")
    callout("즉, 기업이 어떤 목표를 세웠는지, 어떤 환경등급을 받았는지, 실제 온실가스 배출 성과는 어떠한지를 "
            "나누어 살펴보고, 마지막에 <b>환경등급과 배출 성과의 관계</b>를 확인했다.")
    nav_footer(0)


# ──────────────────────────────────────────────────────────────
# 소개 2. 이 프로젝트를 하는 이유
# ──────────────────────────────────────────────────────────────
def page_why():
    header("소개 2", "이 프로젝트를 하는 이유", "같은 환경 데이터도 어떻게 정리하고 비교하느냐에 따라 다르게 보입니다.")
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="card"><span class="step-no">계기</span><h4>같은 데이터, 다른 해석</h4>'
                "<p>환경공학을 이론과 수치로 배우다가, 데이터를 어떻게 정리·비교하느냐에 따라 환경 문제가 다르게 보인다는 점에 흥미를 느꼈다.</p></div>",
                unsafe_allow_html=True)
    c2.markdown('<div class="card"><span class="step-no">경험</span><h4>지표를 함께 봐야 흐름이 보인다</h4>'
                "<p>국가 온실가스 인벤토리 대시보드를 만들며, 배출량 하나보다 연도별 변화·분야별 비중·증감률을 함께 봐야 한다는 것을 경험했다.</p></div>",
                unsafe_allow_html=True)
    c3.markdown('<div class="card"><span class="step-no">목적</span><h4>등급과 실제 성과의 대조</h4>'
                "<p>기업 ESG에도 같은 의문이 생겼고, 평가등급과 실제 배출 성과를 데이터로 검증해 보는 것이 목적이다.</p></div>",
                unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="quote">“ESG 환경등급이 높은 기업은 실제 온실가스 배출 성과도 좋은 기업일까?”</div>',
                unsafe_allow_html=True)
    with st.expander("전체 글 읽기 — 어떻게 접근했나", expanded=False):
      body(
        "환경공학을 전공하면서 환경 문제를 주로 이론과 수치로 배웠지만, 데이터 분석을 배우기 시작하면서 "
        "같은 환경 문제도 데이터를 어떻게 정리하고 비교하느냐에 따라 다르게 볼 수 있다는 점에 흥미를 느꼈다."
        "<br><br>"
        "특히 국가 온실가스 인벤토리 데이터를 활용해 대시보드를 만들어보면서 단순히 배출량 하나를 보는 것보다 "
        "연도별 변화, 분야별 비중, 증감률과 같은 지표를 함께 봐야 전체 흐름을 이해할 수 있다는 것을 경험했다."
        "<br><br>그러면서 기업 ESG에도 비슷한 의문이 생겼다."
        "<br><br>"
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
def grouped_share_bar(cats, series: dict, colors: dict, title: str, key: str, height: int = 380, sub: str = ""):
    labels = [STATUS_KO.get(c, c) for c in cats]
    fig = go.Figure()
    for name, vals in series.items():
        fig.add_bar(name=name, x=labels, y=[vals[c] for c in cats], marker_color=colors[name],
                    text=[f"{vals[c]:.1f}%" for c in cats], textposition="outside",
                    hovertemplate=f"{name}<br>%{{x}}: %{{y:.1f}}%<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    fig.update_yaxes(range=[0, max(float(v.max()) for v in series.values()) * 1.2])
    style_fig(fig, title, height, sub=sub, hide_y=True)
    plot(fig, key)


def stacked_status_bar(counts: pd.DataFrame, pcts: pd.DataFrame, title: str, key: str,
                       name_fn=lambda x: x, sub: str = ""):
    """가로 누적 막대. counts: 기업 수, pcts: 행 기준 비율(%)"""
    labels = [f"{name_fn(i)}  ({int(counts.loc[i].sum()):,}개)" for i in pcts.index]
    fig = go.Figure()
    for status in [s for s in STATUS_ORDER if s in pcts.columns]:
        vals = pcts[status]
        fig.add_bar(name=STATUS_KO[status], y=labels, x=vals, orientation="h", marker_color=STATUS_COLOR[status],
                    marker_line_color="white", marker_line_width=2,
                    text=[f"{v:.0f}%" if v >= 9 else "" for v in vals], textposition="inside",
                    textfont=dict(color=TEXT_ON[status]), insidetextanchor="middle",
                    hovertemplate="%{y}<br>" + STATUS_KO[status] + ": %{x:.1f}%<extra></extra>")
    fig.update_layout(barmode="stack", bargap=0.35, legend_traceorder="normal")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, 100], ticksuffix="%")
    style_fig(fig, title, 96 + 40 * len(labels), horizontal=True, sub=sub)
    plot(fig, key)


def page_sbti():
    header("분석 1 · SBTi", "글로벌·한국 기업은 감축목표를 어디까지 설정했나",
           "SBTi에 등록된 기업의 단기(Near-term)·Net-zero 목표 설정 현황을 보고, 한국 기업을 전체와 비교했습니다.")
    df = load_sbti()
    kr = df[df["location"] == KOREA]
    near, nz = share(df["near_term_status"]), share(df["net_zero_status"])
    t = df[df["near_term_status"] == "Targets set"]["near_term_target_classification"]
    p15 = t.astype(str).str.contains("1.5", na=False).mean() * 100

    st.caption("목표 설정 = Targets set · 설정 약속 = Committed · 약속 철회 = Commitment removed · 미기재 = 상태값 없음")
    tabs = st.tabs(["전체 현황", "지역·산업", "한국과 전체 비교", "산업 구성 보정"])

    with tabs[0]:
        c = st.columns(4)
        kpi(c[0], "SBTi 등록 기업", f"{len(df):,}", "분석 대상 전체")
        kpi(c[1], "단기 목표 설정(Targets set)", pct(near["Targets set"]), "Near-term")
        kpi(c[2], "Scope 1·2 목표 중 1.5°C 수준", pct(p15), "목표 설정 기업 기준")
        kpi(c[3], "Net-zero 목표 설정", pct(nz["Targets set"]), f"단기 대비 {near['Targets set'] - nz['Targets set']:.1f}%p 낮음")
        st.write("")
        grouped_share_bar(STATUS_ORDER, {"단기 목표(Near-term)": near, "Net-zero 목표": nz},
                          {"단기 목표(Near-term)": DEEP, "Net-zero 목표": "#6DB593"}, "단기 목표 vs Net-zero 목표", "sbti_status",
                          sub=f"기업 비율(%) · 단기 목표 설정 {near['Targets set']:.1f}%, Net-zero 목표 설정 {nz['Targets set']:.1f}%")
        callout("SBTi에 참여한 기업은 대부분 단기 감축목표 설정 단계까지 진행했지만, Net-zero 목표 설정까지 이어진 비중은 상대적으로 낮았습니다.")
        caveat("SBTi에 등록된 기업만을 기준으로 한 결과이며 전 세계 모든 기업의 현황이 아닙니다. 이 데이터만으로 실제 감축 진행 정도는 알 수 없습니다. "
               "Net-zero 상태가 비어 있는 기업 중에는 목표 문구에는 있으나 상태 컬럼에 반영되지 않은 예외도 있을 수 있습니다.")

    with tabs[1]:
        reg_n = df["region"].value_counts()
        reg = pd.crosstab(df["region"], df["near_term_status"])
        reg = reg.loc[reg_n.index]
        reg_pct = reg.div(reg.sum(axis=1), axis=0) * 100
        stacked_status_bar(reg, reg_pct, "지역별 단기 목표 상태", "sbti_region", name_fn=ko_region,
                           sub="지역별 기업 비율(%) · 괄호는 등록 기업 수")
        top = df["sector"].value_counts().head(10).index
        sec = pd.crosstab(df[df["sector"].isin(top)]["sector"], df[df["sector"].isin(top)]["near_term_status"]).loc[top]
        sec_pct = sec.div(sec.sum(axis=1), axis=0) * 100
        stacked_status_bar(sec, sec_pct, "기업 수 상위 10개 산업의 단기 목표 상태", "sbti_sector", name_fn=ko_sector,
                           sub="산업별 기업 비율(%) · 괄호는 등록 기업 수")
        top_reg = reg_n.index[0]
        callout(f"기업 수는 {ko_region(top_reg)} 지역이 가장 많고({reg_n.iloc[0] / len(df) * 100:.1f}%), 산업별로도 목표 설정 단계에 차이가 있습니다.")
        caveat("지역·산업별 분포는 SBTi 등록 기업의 구성일 뿐, 특정 지역이 기후대응을 더 잘한다는 뜻이 아닙니다. 표본이 작은 지역은 비율 변동이 큽니다(막대 라벨의 n 참고).")

    with tabs[2]:
        n1, n2 = st.columns(2)
        with n1:
            grouped_share_bar(STATUS_ORDER, {"전체 SBTi": near, f"한국 (n={len(kr)})": share(kr["near_term_status"])},
                              {"전체 SBTi": C_GRAY, f"한국 (n={len(kr)})": C_GREEN}, "단기 목표 상태: 전체 vs 한국", "sbti_kr_near", 380,
                              sub="기업 비율(%)")
        with n2:
            grouped_share_bar(STATUS_ORDER, {"전체 SBTi": nz, f"한국 (n={len(kr)})": share(kr["net_zero_status"])},
                              {"전체 SBTi": C_GRAY, f"한국 (n={len(kr)})": C_GREEN}, "Net-zero 목표 상태: 전체 vs 한국", "sbti_kr_nz", 380,
                              sub="기업 비율(%)")
        kr_n = kr["sector"].value_counts()
        rows = []
        for s in kr_n[kr_n >= 5].index:
            a = kr[kr["sector"] == s]
            b = df[(df["sector"] == s) & (df["location"] != KOREA)]
            ka, kb = (a["near_term_status"] == "Targets set").mean() * 100, (b["near_term_status"] == "Targets set").mean() * 100
            rows.append({"산업": ko_sector(s), "산업(원문)": s, "한국 기업수": len(a), "한국 제외 기업수": len(b),
                         "한국 Targets set(%)": round(ka, 1), "한국 제외 Targets set(%)": round(kb, 1), "차이(%p)": round(ka - kb, 1)})
        tbl = pd.DataFrame(rows).sort_values("차이(%p)", ascending=False)
        fig = go.Figure(go.Bar(
            y=[f"{r['산업']}  (한국 {r['한국 기업수']} / 기타 {r['한국 제외 기업수']})" for _, r in tbl.iterrows()],
            x=tbl["차이(%p)"], orientation="h",
            marker_color=[C_GREEN if v >= 0 else C_AMBER for v in tbl["차이(%p)"]],
            text=[f"{v:+.1f}%p" for v in tbl["차이(%p)"]], textposition="outside",
            hovertemplate="%{y}<br>한국 − 한국 제외: %{x:+.1f}%p<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(ticksuffix="%p")
        fig.add_vline(x=0, line_color=MUTED, line_width=1)
        style_fig(fig, "한국 기업 5개 이상 산업의 목표 설정 비율 차이", 90 + 46 * len(tbl), horizontal=True,
                  sub="한국 기업의 목표 설정 비율 − 한국 제외 기업의 비율(%p) · 초록은 한국이 더 높은 산업")
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
        fig.add_bar(name="전체 SBTi", y=[ko_sector(s) for s in top10], x=[gs.get(s, 0) for s in top10], orientation="h", marker_color=C_GRAY,
                    text=[f"{gs.get(s, 0):.1f}%" for s in top10], textposition="outside")
        fig.add_bar(name="한국", y=[ko_sector(s) for s in top10], x=[ks[s] for s in top10], orientation="h", marker_color=C_GREEN,
                    text=[f"{ks[s]:.1f}%" for s in top10], textposition="outside")
        fig.update_layout(barmode="group", bargap=0.3)
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(ticksuffix="%")
        style_fig(fig, "한국 상위 10개 산업의 비중: 한국 vs 전체", 620, horizontal=True,
                  sub="기업 수 기준 산업 비중(%) · 한국은 자동차·부품, 금융 비중이 높음")
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
        mode = st.radio("보기 방식", ["3단계로 묶어 보기", "6개 등급 그대로"], horizontal=True, key="kcgs_mode")
        xs = [str(y) for y in dist.index]
        fig = go.Figure()
        if mode.startswith("3"):
            tiers = [("우수 (A+·A)", ["A+", "A"], DEEP, "white"), ("보통 (B+·B)", ["B+", "B"], "#6DB593", INK),
                     ("취약 (C·D)", ["C", "D"], "#D3DDD7", INK)]
            for name, gl, col, tc in tiers:
                vals = dist[gl].sum(axis=1)
                fig.add_bar(name=name, x=xs, y=vals, marker_color=col, marker_line_color="white", marker_line_width=2,
                            text=[f"{v:.0f}%" for v in vals], textposition="inside", textfont=dict(color=tc, size=15),
                            hovertemplate=f"{name}<br>%{{x}}년: %{{y:.1f}}%<extra></extra>")
            sub = "각 연도에 평가된 기업의 비율(%) · 우수 = A+·A, 보통 = B+·B, 취약 = C·D"
        else:
            for g in GRADES:
                fig.add_bar(name=g, x=xs, y=dist[g], marker_color=GRADE_COLOR[g], marker_line_color="white", marker_line_width=2,
                            text=[f"{v:.0f}%" if v >= 7 else "" for v in dist[g]], textposition="inside",
                            textfont=dict(color="white" if g in ("A+", "A", "B+") else INK, size=14),
                            hovertemplate=f"{g}<br>%{{x}}년: %{{y:.1f}}%<extra></extra>")
            sub = "각 연도에 평가된 기업의 비율(%) · 7% 미만 구간은 숫자를 생략했습니다"
        fig.update_layout(barmode="stack", bargap=0.3, legend_traceorder="normal")
        fig.update_yaxes(range=[0, 100], ticksuffix="%")
        style_fig(fig, "연도별 환경등급 분포", 440, sub=sub)
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
                        text=[f"{v:.0f}%" if v >= 5 else "" for v in vals], textposition="inside",
                        textfont=dict(color=INK if name == "유지" else "white", size=15),
                        hovertemplate=f"{name}<br>%{{x}}: %{{y:.1f}}%<extra></extra>")
        fig.update_layout(barmode="stack", bargap=0.3, legend_traceorder="normal")
        fig.update_yaxes(range=[0, 100], ticksuffix="%")
        style_fig(fig, "같은 기업의 환경등급 이동", 440, sub="두 해 모두 평가된 같은 기업의 상향·유지·하향 비율(%) · n은 비교 기업 수")
        plot(fig, "kcgs_change")
        callout("2021→2022년에는 하향 기업이 많았고, 2022→2023년에는 상향이 크게 늘었습니다. 2023년 이후에는 등급을 유지하는 기업이 대부분입니다.")
        caveat("이 변화는 KCGS 평가등급의 변화입니다. 기업의 실제 환경성과가 같은 수준으로 변했다는 뜻은 아니며, 평가기준·모형 변화의 영향도 있을 수 있습니다.")

    with tabs[2]:
        label = st.selectbox("비교 기간", [f"{a}→{b}" for a, b in PERIODS], index=1, key="kcgs_period")
        y0, y1 = (int(x) for x in label.split("→"))
        m = kcgs_change(k, y0, y1)
        ct = pd.crosstab(m[f"환경_{y0}"], m[f"환경_{y1}"]).reindex(index=GRADES, columns=GRADES, fill_value=0)
        fig = px.imshow(ct, text_auto=True, aspect="auto", color_continuous_scale=["#F1F7F3", "#86CBA5", DEEP],
                        labels=dict(x=f"{y1}년 등급", y=f"{y0}년 등급", color="기업 수"))
        fig.update_coloraxes(showscale=False)
        style_fig(fig, f"{y0}년 → {y1}년 환경등급 이동표", 480, sub="세로: 이전 연도 등급 · 가로: 다음 연도 등급 · 숫자는 기업 수")
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
        yv = bal.values / 1e6
        fig = go.Figure(go.Scatter(
            x=[str(y) for y in bal.index], y=yv, mode="lines+markers+text",
            line=dict(color=C_GREEN, width=3), marker=dict(size=11, color="white", line=dict(color=C_GREEN, width=3)),
            text=[f"{v:.1f}" for v in yv], textposition="top center", textfont=dict(size=13, color=INK),
            fill="tozeroy", fillcolor="rgba(27,127,92,0.08)",
            hovertemplate="%{x}년: %{y:.1f}M tCO₂eq<extra></extra>"))
        fig.update_yaxes(range=[yv.min() * 0.85, yv.max() * 1.08])
        style_fig(fig, f"균형패널 {n_bal}개 기업의 총배출량", 380,
                  sub="단위: 백만 tCO₂eq · 세로축이 0에서 시작하지 않아 변화폭이 크게 보입니다", hide_y=True)
        plot(fig, "gir_total")
    with b:
        cnt = g["대상년도"].value_counts().sort_index()
        fig = go.Figure(go.Bar(x=[str(y) for y in cnt.index], y=cnt.values, marker_color=C_GRAY,
                               text=cnt.values, textposition="outside"))
        fig.update_yaxes(range=[0, cnt.max() * 1.15])
        style_fig(fig, "연도별 관측 기업 수", 380, sub="단위: 기업 수", hide_y=True)
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
            dd = df.dropna(subset=["배출집약도"])
            dd = dd[dd["배출집약도"] > 0]
            q = dd.groupby("환경")["배출집약도"].quantile([0.25, 0.5, 0.75]).unstack().reindex(GRADES)
            lab = [f"{g}<br>n={int(gs.loc[g, 'n']):,}" if pd.notna(gs.loc[g, "n"]) else g for g in GRADES]
            fig = go.Figure()
            for g, l in zip(GRADES, lab):
                if pd.notna(q.loc[g, 0.5]):
                    fig.add_trace(go.Scatter(x=[l, l], y=[q.loc[g, 0.25], q.loc[g, 0.75]], mode="lines",
                                             line=dict(color="#A9D3BC", width=22), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=lab, y=q[0.5], mode="markers+text",
                                     marker=dict(size=15, color=DEEP, line=dict(color="white", width=2)),
                                     text=[f"{v:,.0f}" if pd.notna(v) else "" for v in q[0.5]], textposition="middle right",
                                     textfont=dict(size=15, color=DEEP), showlegend=False,
                                     customdata=np.stack([q[0.25], q[0.75]], axis=-1),
                                     hovertemplate="중앙값 %{y:,.1f}<br>중간 50%: %{customdata[0]:,.1f} ~ %{customdata[1]:,.1f}<extra></extra>"))
            fig.update_yaxes(type="log", tickvals=[1, 10, 100, 1000, 10000], ticktext=["1", "10", "100", "1천", "1만"])
            fig.update_xaxes(title_text="환경등급 (n = 기업연도 수)")
            style_fig(fig, "등급별 배출집약도", 420,
                      sub="● 중앙값 · 연한 막대 = 중간 50% 기업(25~75%) · 세로축은 로그 · 단위: tCO₂eq / 매출 10억원")
            plot(fig, "cmb_int")
        with b:
            fig = go.Figure(go.Bar(
                x=[f"{g}<br>n={int(es.loc[g, 'n']):,}" if pd.notna(es.loc[g, "n"]) else g for g in GRADES],
                y=es["med"], marker_color=[GRADE_COLOR[x] for x in GRADES],
                marker_line_color="#8DB9A2", marker_line_width=1,
                text=[f"{v:,.0f}" for v in es["med"]], textposition="outside",
                customdata=es["n"], hovertemplate="%{x}등급<br>중앙값 %{y:,.0f} tCO₂eq<br>기업연도 %{customdata:,}<extra></extra>"))
            fig.update_yaxes(range=[0, np.nanmax(es["med"]) * 1.2])
            fig.update_xaxes(title_text="환경등급")
            style_fig(fig, "등급별 절대배출량 (중앙값)", 420, sub="단위: tCO₂eq · 막대 아래 n = 기업연도 수", hide_y=True)
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
        style_fig(fig, "등급 변화 방향별 배출집약도 증감률", 420, sub="배출집약도 증감률의 중앙값(%) · 막대 아래 n은 비교 건수", hide_y=True)
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
    P(page_why, title="이 프로젝트를 하는 이유", icon=":material/grass:", url_path="why"),
    P(page_overview, title="프로젝트 개요 및 순서", icon=":material/nature:", url_path="overview"),
    P(page_sbti, title="분석 1 · SBTi 감축목표", icon=":material/park:", url_path="sbti"),
    P(page_kcgs, title="분석 2 · KCGS 환경등급", icon=":material/forest:", url_path="kcgs"),
    P(page_gir, title="분석 3 · GIR 배출량", icon=":material/yard:", url_path="gir"),
    P(page_combined, title="분석 4 · 등급과 배출 성과", icon=":material/nature_people:", url_path="combined"),
    P(page_final, title="최종 분석", icon=":material/local_florist:", url_path="final"),
    P(page_refs, title="참고 자료 · 데이터", icon=":material/spa:", url_path="refs"),
]
ORDER = PAGE_LIST
NAV = {"소개": PAGE_LIST[0:3], "분석": PAGE_LIST[3:8], "부록": PAGE_LIST[8:9]}

st.sidebar.caption(f"{PROJECT_TITLE}\n\n{PROJECT_SUBTITLE}")
pg = st.navigation(NAV)
pg.run()
