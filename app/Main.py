import streamlit as st
import psycopg2
import numpy as np
import regex as re
from sentence_transformers import SentenceTransformer
from collections import Counter
from pgvector.psycopg2 import register_vector
import html
import time
from datetime import datetime
from bs4 import BeautifulSoup
import markdown

from dotenv import load_dotenv
import os

load_dotenv()  # Load variables from .env file

PASSWORD = os.getenv("DB_PASSWORD")
USER = os.getenv("DB_USER")
HOST = os.getenv("DB_HOST")
PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

SUPABASE_DB_URL = f"postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB_NAME}"

# 页面配置
st.set_page_config(
    page_title="政策文档智能搜索系统",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)


# 初始化向量模型
@st.cache_resource
def load_model():
    return SentenceTransformer("moka-ai/m3e-base")


model = load_model()


# 连接 Supabase
def get_db_connection():
    try:
        conn = psycopg2.connect(SUPABASE_DB_URL)
        register_vector(conn)  # 注册 `pgvector`
        return conn
    except Exception as e:
        st.error(f"数据库连接失败: {str(e)}")
        return None


# 侧边栏筛选器
# st.sidebar.title("搜索设置")


# 缓存筛选选项
@st.cache_data(ttl=3600)  # 1小时缓存
def load_filters():
    """加载可用的分类筛选数据"""
    conn = get_db_connection()
    if not conn:
        return [], [], []

    cursor = conn.cursor()

    cursor.execute("SELECT DISTINCT location FROM test_documents")
    locations = [row[0] for row in cursor.fetchall() if row[0]]

    cursor.execute("SELECT DISTINCT category FROM test_documents")
    categories = [row[0] for row in cursor.fetchall() if row[0]]

    cursor.execute("SELECT DISTINCT publish_date FROM test_documents")
    dates = [str(row[0]) for row in cursor.fetchall() if row[0]]

    conn.close()
    return locations, categories, dates


# 缓存文件列表
@st.cache_data(ttl=3600)  # 1小时缓存
def load_files():
    """加载文件列表"""
    conn = get_db_connection()
    if not conn:
        return []

    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT DISTINCT file_name, category, location, publish_date 
        FROM test_documents 
        ORDER BY file_name
    """
    )
    files = [(row[0], row[1], row[2], row[3]) for row in cursor.fetchall()]
    conn.close()
    return files


# 通用工具函数
def contains_number(text):
    """判断文本是否包含数字/年份/金额/百分比"""
    number_pattern = r"(\d{4}年|\$\d+(?:,\d+)*|\d+\.\d+%|\d+)"
    return bool(re.search(number_pattern, text))


def highlight_keywords(text, query):
    """高亮关键词"""
    query = html.escape(query)
    text = html.escape(text)

    keywords = query.split()
    for word in keywords:
        if word.strip():
            pattern = rf"({re.escape(word)})"
            text = re.sub(
                pattern,
                r'<span style="color:red; font-weight:bold;">\1</span>',
                text,
                flags=re.IGNORECASE,
            )
    return text


# 获取上下文
def get_context(segment_id, max_context=3):
    """获取段落的上下文（前后几个段落）"""
    conn = get_db_connection()
    if not conn:
        return None

    cursor = conn.cursor()

    # 获取当前段落所属文档ID
    cursor.execute(
        """
        SELECT document_id FROM test_document_relations WHERE segment_id = %s
    """,
        (segment_id,),
    )
    row = cursor.fetchone()
    if not row:
        conn.close()
        return None

    document_id = row[0]

    # 获取当前段落的序号
    cursor.execute(
        """
        SELECT segment_position FROM test_document_relations 
        WHERE document_id = %s AND segment_id = %s
    """,
        (document_id, segment_id),
    )
    position_row = cursor.fetchone()
    current_position = position_row[0] if position_row else 0

    # 获取上下文段落
    cursor.execute(
        """
        SELECT r.segment_position, s.id, s.segment_text 
        FROM test_document_relations r
        JOIN test_document_segments s ON r.segment_id = s.id
        WHERE r.document_id = %s 
        AND r.segment_position BETWEEN %s AND %s
        ORDER BY r.segment_position
    """,
        (
            document_id,
            max(0, current_position - max_context),
            current_position + max_context,
        ),
    )

    context = []
    for pos, seg_id, text in cursor.fetchall():
        context.append(
            {
                "position": pos,
                "segment_id": seg_id,
                "text": text,
                "is_current": seg_id == segment_id,
            }
        )

    conn.close()
    return context


def search(query, selected_files, selected_location, selected_category, selected_date):
    """在 `pgvector` 进行向量搜索，支持多维度过滤和语义搜索"""
    # 处理空查询的情况
    if not query.strip():
        return []

    # 分析查询关键词
    keywords = query.split()

    # 创建查询向量
    query_vector = model.encode([query], convert_to_numpy=True)[0]

    conn = get_db_connection()
    cursor = conn.cursor()

    conditions = []
    params = [query_vector]

    if selected_files and selected_files != all_files:
        conditions.append(
            f"test_documents.file_name IN ({', '.join(['%s'] * len(selected_files))})"
        )
        params.extend(selected_files)

    if selected_location and selected_location != locations:
        placeholders = ", ".join(["%s"] * len(selected_location))
        conditions.append(f"test_documents.location IN ({placeholders})")
        params.extend(selected_location)

    if selected_category:
        conditions.append("test_documents.category = %s")
        params.append(selected_category)

    if selected_date:
        conditions.append("test_documents.publish_date = %s")
        params.append(selected_date)

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # 优化查询以包含更多元数据和相关性信息
    sql_query = f"""
        SELECT 
            test_documents.file_name, 
            test_documents.category,
            test_documents.location,
            test_documents.publish_date,
            test_document_segments.id, 
            test_document_segments.segment_text,
            test_document_segments.vector_embedding <=> %s AS distance
        FROM test_document_relations
        JOIN test_documents ON test_document_relations.document_id = test_documents.id
        JOIN test_document_segments ON test_document_relations.segment_id = test_document_segments.id
        WHERE {where_clause}
        ORDER BY distance ASC
        LIMIT 100;  -- 增加初始获取量以改善多样性
    """

    cursor.execute(sql_query, params)
    results = cursor.fetchall()

    doc_scores = {}
    doc_texts = {}
    doc_metadata = {}

    # 更复杂的相关性计算
    for row in results:
        (
            file_name,
            category,
            location,
            publish_date,
            segment_id,
            text_segment,
            distance,
        ) = row

        # 改进的相关性评分
        base_score = 1 / (1 + distance)  # 向量匹配得分

        # 加权因子
        keyword_match_bonus = 1.0
        for keyword in keywords:
            if re.search(rf"\b{re.escape(keyword)}\b", text_segment, re.IGNORECASE):
                keyword_match_bonus += 0.2  # 增加关键词匹配权重

        number_bonus = 1.2 if contains_number(text_segment) else 1.0  # 数值加权
        length_penalty = min(
            1.0, max(0.8, len(text_segment) / 500)
        )  # 长度惩罚，避免过长文本

        # 综合评分
        final_score = base_score * number_bonus * keyword_match_bonus * length_penalty

        if file_name not in doc_scores:
            doc_scores[file_name] = 0
            doc_texts[file_name] = []
            doc_metadata[file_name] = {
                "category": category,
                "location": location,
                "publish_date": publish_date,
            }

        doc_scores[file_name] += final_score
        doc_texts[file_name].append((segment_id, text_segment, final_score))

    cursor.close()
    conn.close()

    # 选出排名前 10 的最佳文档
    top_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:10]

    final_results = []
    for file_name, _ in top_docs:
        # 获取文档的元数据
        metadata = doc_metadata[file_name]

        # 选择最相关的段落，但确保它们不是重复内容
        paragraphs = doc_texts[file_name]
        paragraphs.sort(key=lambda x: x[2], reverse=True)

        # 去重：确保不选择内容过于相似的段落
        unique_paragraphs = []
        used_content = set()

        for segment_id, text, score in paragraphs:
            # 简单的去重机制：检查前50个字符是否重复
            text_start = text[:50].lower()
            if text_start not in used_content and len(unique_paragraphs) < 3:
                unique_paragraphs.append((segment_id, text, score))
                used_content.add(text_start)

        final_results.append(
            {
                "file_name": file_name,
                "category": metadata["category"],
                "location": metadata["location"],
                "publish_date": metadata["publish_date"],
                "best_paragraphs": unique_paragraphs,
            }
        )

    return final_results


import html


def clean_to_plain_text_with_titles(text, query=None):
    # 1. Remove LaTeX syntax ($...$)
    text = re.sub(r"\$.*?\$", "", text)

    # 2. Replace markdown headers (e.g., `# Title` → `Title`) and preserve titles/subtitles
    text = re.sub(r"^(#{1,6})\s*(.*)", lambda m: f"{m.group(2)}", text)

    # 3. Convert markdown line breaks (like `\n`) to `<br>` to preserve the structure
    text = text.replace("\n", " <br> ")

    # 4. Remove HTML tags (if any)
    soup = BeautifulSoup(text, "html.parser")
    plain_text = soup.get_text(separator=" ")

    # 5. Unescape HTML entities (like `&nbsp;` to spaces)
    plain_text = html.unescape(plain_text)

    # 6. Optional: Highlight query keywords in plain text
    if query:
        for word in query.split():
            if word.strip():
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                plain_text = pattern.sub(lambda m: f"**{m.group(0)}**", plain_text)

    return plain_text.strip()


# 显示搜索结果
def display_search_results(query, results):
    # 结果面板
    st.subheader("📁 搜索结果")

    # 分类汇总
    categories = Counter([doc["category"] for doc in results if doc["category"]])
    locations = Counter([doc["location"] for doc in results if doc["location"]])

    # 在搜索结果之上显示搜索摘要
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**按类别分布:**")
        for cat, count in categories.most_common():
            st.markdown(f"- {cat}: {count}个文档")

    with col2:
        st.markdown("**按地区分布:**")
        for loc, count in locations.most_common():
            st.markdown(f"- {loc}: {count}个文档")

    # 显示结果列表
    for idx, doc in enumerate(results):
        file_name = doc["file_name"]
        category = doc["category"] if doc["category"] else "未分类"
        location = doc["location"] if doc["location"] else "未知地区"
        publish_date = doc["publish_date"] if doc["publish_date"] else "未知日期"

        with st.expander(
            f"📄 {file_name} [{category} - {location} - {publish_date}]",
            expanded=idx == 0,
        ):
            # 为每个搜索结果添加相关度信息
            st.markdown("##### 最相关段落")

            # 显示段落
            for i, (segment_id, paragraph, score) in enumerate(doc["best_paragraphs"]):
                # Clean and render as plain text
                plain_text = clean_to_plain_text_with_titles(paragraph, query)

                # Display with a simple card style
                st.markdown(
                    f"""
                <div style='background-color:{"#ffebcc" if i==0 else "#f9f9f9"}; 
                            padding:15px; 
                            border-radius:8px; 
                            margin-bottom:10px;
                            border-left:4px solid {"#ff9900" if i==0 else "#cccccc"};'>
                    <div style='display:flex; justify-content:space-between; margin-bottom:5px;'>
                        <span style='font-weight:bold;'>{"🔥 最佳匹配" if i==0 else f"匹配 #{i+1}"}</span>
                        <span style='color:#666;'>相关度: {score:.2f}</span>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

                # Show the cleaned text
                st.markdown(f"> {plain_text}")

                st.markdown("</div>", unsafe_allow_html=True)

                # if "context_visible" not in st.session_state:
                #     st.session_state.context_visible = {}

                # # 替换按钮处理逻辑
                # context_key = f"context_{segment_id}"
                # if st.button(f"📑 查看上下文", key=f"btn_{context_key}"):
                #     # 切换显示状态
                #     st.session_state.context_visible[context_key] = not st.session_state.context_visible.get(context_key, False)

                # # 根据状态显示上下文
                # if st.session_state.context_visible.get(context_key, False):
                #     context_paragraphs = get_context(segment_id)
                #     if context_paragraphs:
                #         st.markdown("##### 上下文段落")
                #         for ctx in context_paragraphs:
                #             is_current = ctx["is_current"]
                #             ctx_text = highlight_keywords(ctx["text"], query) if is_current else ctx["text"]

                #             st.markdown(f"""
                #             <div style='background-color:{"#e6f7ff" if is_current else "#ffffff"};
                #                        padding:10px;
                #                        border-radius:5px;
                #                        margin-bottom:5px;
                #                        border-left:3px solid {"#1890ff" if is_current else "#e8e8e8"};'>
                #                 {ctx_text}
                #             </div>
                #             """, unsafe_allow_html=True)
                #     else:
                #         st.info("无法获取上下文内容")

                # 提供上下文查看按钮
                # if st.button(f"📑 查看上下文", key=f"context_{segment_id}"):
                #     context_paragraphs = get_context(segment_id)
                #     if context_paragraphs:
                #         st.markdown("##### 上下文段落")
                #         for ctx in context_paragraphs:
                #             is_current = ctx["is_current"]
                #             ctx_text = highlight_keywords(ctx["text"], query) if is_current else ctx["text"]

                #             st.markdown(f"""
                #             <div style='background-color:{"#e6f7ff" if is_current else "#ffffff"};
                #                        padding:10px;
                #                        border-radius:5px;
                #                        margin-bottom:5px;
                #                        border-left:3px solid {"#1890ff" if is_current else "#e8e8e8"};'>
                #                 {ctx_text}
                #             </div>
                #             """, unsafe_allow_html=True)
                #     else:
                #         st.info("无法获取上下文内容")

            # 添加导出选项
            # col1, col2 = st.columns(2)
            # with col1:
            #     if st.button("📋 复制到剪贴板", key=f"copy_{file_name}"):
            #         extract_text = "\n\n".join([para for _, para, _ in doc["best_paragraphs"]])
            #         st.code(extract_text)
            #         st.success("内容已复制，请使用Ctrl+C复制上方代码框中的文本")

            # with col2:
            #     # 添加查看原文选项
            #     st.markdown(f"[📖 查看完整文档](#{file_name})")


# 添加数据反馈功能
def feedback_handler():
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = set()

    def give_feedback(segment_id, rating):
        if segment_id in st.session_state.feedback_given:
            return

        conn = get_db_connection()
        if not conn:
            st.error("无法连接到数据库提交反馈")
            return

        cursor = conn.cursor()

        try:
            # 检查是否已有反馈表，如果没有则创建
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id SERIAL PRIMARY KEY,
                    segment_id INTEGER NOT NULL,
                    rating INTEGER NOT NULL, -- 1-5分
                    feedback_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            conn.commit()

            # 记录反馈
            cursor.execute(
                """
                INSERT INTO user_feedback (segment_id, rating)
                VALUES (%s, %s)
            """,
                (segment_id, rating),
            )
            conn.commit()

            st.session_state.feedback_given.add(segment_id)
            st.success("谢谢您的反馈！")
        except Exception as e:
            st.error(f"提交反馈失败: {str(e)}")
        finally:
            conn.close()

    return give_feedback


# 添加导出文档功能
def generate_report(query, results):
    """生成可导出的报告"""
    report = f"""# 搜索报告："{query}"
生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 搜索结果摘要
共找到 {len(results)} 个相关文档

"""
    # 添加每个文档的内容
    for i, doc in enumerate(results):
        report += f"""
### {i+1}. {doc['file_name']}
- 分类: {doc['category'] if doc['category'] else '未分类'}
- 地区: {doc['location'] if doc['location'] else '未知'}
- 发布日期: {doc['publish_date'] if doc['publish_date'] else '未知'}

#### 相关段落:
"""
        for j, (_, para, score) in enumerate(doc["best_paragraphs"]):
            report += f"""
**段落 {j+1}** (相关度: {score:.2f})
{para}

"""

    return report


def save_search_history_to_db(query_text, timestamp):
    conn = get_db_connection()
    if not conn:
        return

    cursor = conn.cursor()

    try:
        # 如果表不存在，则创建表
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_queries (
                id SERIAL PRIMARY KEY,
                query_text TEXT NOT NULL,
                query_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        conn.commit()

        # 插入查询记录
        cursor.execute(
            """
            INSERT INTO user_queries (query_text, query_time)
            VALUES (%s, %s)
        """,
            (query_text, timestamp),
        )
        conn.commit()
    except Exception as e:
        st.error(f"无法保存查询历史: {str(e)}")
    finally:
        conn.close()


# 配置全局参数
# st.sidebar.markdown("### 应用设置")
# # 允许用户调整搜索设置
# max_results = st.sidebar.slider("最大结果数量", 5, 20, 10)
# relevance_threshold = st.sidebar.slider("最低相关度阈值", 0.1, 0.9, 0.5)
max_results = 10
relevance_threshold = 0.5

# 运行主应用
# 加载数据
locations, categories, dates = load_filters()
files_data = load_files()
all_files = [f[0] for f in files_data]

# 标题区域
st.title("📘 政策文档智能搜索系统（展示版）")

st.write(
    """
    - 2024年12月全国储能政策 + 136号文件 + 1个标准
    - 混合使用关键词匹配和语义搜索，找到最相关的段落
    - 对于包含数字、年份、金额等的文本，会有额外的加权以提高相关性
    - **后续服务器接入之后，可以支持全文访问**
"""
)

# 搜索栏
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input("🔍 请输入搜索关键词", key="search_box")
with col2:
    search_button = st.button("开始搜索", use_container_width=True)

# 高级筛选区
with st.expander("高级筛选选项", expanded=False):
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_locations = st.multiselect("🌍 选择地点", locations, default=locations)

    with col2:
        selected_category = st.selectbox("📂 选择类别", [""] + categories, index=0)

    with col3:
        selected_date = st.selectbox("📅 选择时间", [""] + dates, index=0)

# 搜索历史记录
if "search_history" not in st.session_state:
    st.session_state.search_history = []

# 处理搜索请求
if search_button and query:
    start_time = time.time()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ✅ 保存历史到数据库
    save_search_history_to_db(query, timestamp)

    with st.spinner("正在搜索相关内容..."):
        best_match_docs = search(
            query, all_files, selected_locations, selected_category, selected_date
        )

        if best_match_docs:
            end_time = time.time()
            st.success(
                f"✅ 找到 {len(best_match_docs)} 个相关文档 (用时 {end_time - start_time:.2f} 秒)"
            )
            display_search_results(query, best_match_docs)
        else:
            st.warning("❌ 未找到相关内容，请尝试不同的关键词。")


# if search_button and query:
#     start_time = time.time()

#     # 添加到搜索历史
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     st.session_state.search_history.insert(0, {
#         "query": query,
#         "timestamp": timestamp
#     })

#     # 最多保留10条历史记录
#     if len(st.session_state.search_history) > 10:
#         st.session_state.search_history = st.session_state.search_history[:10]

#     with st.spinner("正在搜索相关内容..."):
#         best_match_docs = search(query, all_files, selected_locations, selected_category, selected_date)

#         # 显示搜索结果
#         if best_match_docs:
#             end_time = time.time()
#             st.success(f"✅ 找到 {len(best_match_docs)} 个相关文档 (用时 {end_time - start_time:.2f} 秒)")

#             # 显示结果
#             display_search_results(query, best_match_docs)
#         else:
#             st.warning("❌ 未找到相关内容，请尝试不同的关键词。")
# 显示搜索历史
# if st.session_state.search_history:
#     st.sidebar.subheader("搜索历史")
#     for idx, item in enumerate(st.session_state.search_history):
#         if st.sidebar.button(f"{item['query']} ({item['timestamp']})", key=f"history_{idx}"):
#             # 点击历史记录时，填充搜索框并执行搜索
#             st.session_state.search_box = item['query']
#             st.experimental_rerun()


# 页脚
st.markdown("---")
st.markdown("📘 **政策文档智能搜索系统** | 基于语义向量检索")
