import sqlite3
import streamlit as st
import plotly.figure_factory as ff
import plotly.express as px


def fetchall_dict(cur):
    header = [d[0] for d in cur.description]
    return [dict(zip(header, r)) for r in cur.fetchall()]


connection = sqlite3.connect('identifier.sqlite')
cursor = connection.cursor()

st.subheader('Monthly Histogram')
cursor.execute(
    """
        SELECT strftime('%Y-%m',created_at) month, count(1) freq
        FROM pooled_sample_tweets
        GROUP BY month
        ORDER BY month;
    """)

res = fetchall_dict(cursor)
fig = px.bar(res, x='month', y='freq', range_y=(0, 3500))
st.plotly_chart(fig, use_container_width=True)

st.subheader('Weekly Histogram')
cursor.execute(
    """
        SELECT strftime('%Y-%W',created_at) week, count(1) freq
        FROM pooled_sample_tweets
        GROUP BY week
        ORDER BY week;
    """)

res = fetchall_dict(cursor)
fig = px.bar(res, x='week', y='freq', range_y=(0, 1000))
st.plotly_chart(fig, use_container_width=True)

st.subheader('Daily Histogram')
cursor.execute(
    """
        SELECT strftime('%Y-%m-%d',created_at) day, count(1) freq
        FROM pooled_sample_tweets
        GROUP BY day
        ORDER BY day;
    """)

res = fetchall_dict(cursor)
fig = px.bar(res, x='day', y='freq', range_y=(0, 320))
st.plotly_chart(fig, use_container_width=True)

# st.bar_chart([r['freq'] for r in res])
st.write(res)
