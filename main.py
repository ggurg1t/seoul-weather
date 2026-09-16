import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/seoul.csv"

st.set_page_config(
    page_title="서울 100년 기온 변화",
    page_icon="🌡️",
    layout="centered",
)


@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_URL)
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    return df


@st.cache_data(show_spinner=False)
def build_yearly(df, min_days):
    # 한 해의 관측일이 너무 적으면(첫 해·마지막 해 등) 평균이 왜곡될 수 있어 제외
    counts = df.groupby("연도")["평균기온"].count()
    valid_years = counts[counts >= min_days].index
    yearly = (
        df[df["연도"].isin(valid_years)]
        .groupby("연도")["평균기온"]
        .mean()
        .reset_index()
        .rename(columns={"평균기온": "연평균기온"})
    )
    return yearly


def main():
    st.title("🌡️ 서울, 100년의 기온 변화")
    st.write(
        "서울 기상관측 데이터를 이용해 연평균 기온이 지난 100여 년 동안 "
        "어떻게 변해 왔는지 살펴보는 페이지입니다."
    )

    with st.spinner("데이터를 불러오는 중이에요..."):
        df = load_data()

    yearly = build_yearly(df, min_days=300)

    min_year = int(yearly["연도"].min())
    max_year = int(yearly["연도"].max())

    st.caption(f"관측 기간: {min_year}년 ~ {max_year}년 (관측 일수가 적은 해는 제외했어요)")

    year_range = st.slider(
        "살펴볼 기간을 골라보세요",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
    )

    view = yearly[(yearly["연도"] >= year_range[0]) & (yearly["연도"] <= year_range[1])]

    # 추세선 (선형회귀)
    coeffs = np.polyfit(view["연도"], view["연평균기온"], 1)
    trend = np.poly1d(coeffs)(view["연도"])
    slope_per_decade = coeffs[0] * 10

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=view["연도"],
            y=view["연평균기온"],
            mode="lines+markers",
            name="연평균 기온",
            line=dict(color="#E39A12", width=2),
            marker=dict(size=4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=view["연도"],
            y=trend,
            mode="lines",
            name="추세선",
            line=dict(color="#5F9E6E", width=3, dash="dash"),
        )
    )
    fig.update_layout(
        xaxis_title="연도",
        yaxis_title="연평균 기온 (℃)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="#FFFDF4",
        paper_bgcolor="#FFFDF4",
    )

    st.plotly_chart(fig, use_container_width=True)

    first_avg = view["연평균기온"].iloc[0]
    last_avg = view["연평균기온"].iloc[-1]
    diff = last_avg - first_avg

    col1, col2, col3 = st.columns(3)
    col1.metric(f"{int(view['연도'].iloc[0])}년 연평균", f"{first_avg:.1f} ℃")
    col2.metric(f"{int(view['연도'].iloc[-1])}년 연평균", f"{last_avg:.1f} ℃", f"{diff:+.1f} ℃")
    col3.metric("10년당 상승폭 (추세선 기준)", f"{slope_per_decade:+.2f} ℃")

    st.markdown(
        f"""
선택한 기간({year_range[0]}년 ~ {year_range[1]}년) 동안 서울의 연평균 기온은
해마다 오르내림을 반복하면서도, 초록색 점선으로 표시한 추세선을 보면
10년마다 평균 **{slope_per_decade:+.2f}℃** 정도 변해 온 것으로 나타납니다.
"""
    )

    with st.expander("연도별 원자료 보기"):
        st.dataframe(
            view.rename(columns={"연도": "연도", "연평균기온": "연평균 기온(℃)"}),
            use_container_width=True,
            hide_index=True,
        )

    st.caption("자료 출처: 기상청 서울(종로구) 관측소 일별 기온 데이터")


if __name__ == "__main__":
    main()
