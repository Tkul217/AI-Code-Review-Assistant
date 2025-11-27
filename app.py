# AI-Code Review Assistant for ForteBank AI Hackathon
# Task 5: AI-Code Review Assistant
# Implemented in Python with Google Gemini API for AI analysis
# Integration with GitLab API for automatic MR reviews via webhooks
# Streamlit interface for manual reviews and demos

import os
import time
import json
import re
import requests
from flask import Flask, request, jsonify
import streamlit as st
import gitlab  # python-gitlab for easier API interaction
import google.generativeai as genai
from dotenv import load_dotenv  # For loading .env

# Load environment variables
load_dotenv()
GITLAB_TOKEN = os.getenv('GITLAB_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GITLAB_URL = os.getenv('GITLAB_URL', 'https://gitlab.com')

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')  # Fast model for <5min analysis

# GitLab client
gl = gitlab.Gitlab(GITLAB_URL, private_token=GITLAB_TOKEN)

# Prompt template for AI code review (optimized for relevance, accuracy, best practices)
REVIEW_PROMPT = """
Ты опытный senior разработчик, выполняющий code review для GitLab Merge Request.
Тщательно проанализируй предоставленный diff кода. Проверь:
- Соответствие стандартам кодирования (например, PEP8 для Python, общие best practices для других языков).
- Потенциальные баги, проблемы безопасности, проблемы производительности.
- Корректность относительно описанной функциональности (используй название и описание MR как контекст).
- Предложения по улучшению (читаемость, эффективность, модульность).

ВАЖНО: Отвечай ТОЛЬКО на РУССКОМ языке. Все комментарии, рекомендации и описания должны быть на русском.

Выведи валидный JSON объект со следующей структурой:
{{
  "summary": "Краткая сводка ревью на русском языке (1-2 параграфа).",
  "recommendation": "merge or needs fixes or reject",
  "general_comments": ["Список общих рекомендаций на русском языке."],
  "line_comments": [
    {{"path": "file/path.py", "line": 10, "comment": "Рекомендация для этой строки на русском языке."}}
  ]
}}

Название MR: {title}
Описание MR: {description}
Код Diff:
{diff}
"""


# Function to clean and validate JSON string
def clean_json(s):
    """Clean JSON string from markdown and whitespace"""
    s = s.strip()

    # Remove markdown code blocks
    s = re.sub(r'^```json\s*|\s*```$', '', s, flags=re.MULTILINE)
    s = s.strip()

    # Extract JSON object if it's embedded in text
    match = re.search(r'\{.*\}', s, re.DOTALL)
    if match:
        s = match.group(0)

    return s


# Function to perform AI review
def ai_review_mr(project_id, mr_iid):
    start_time = time.time()

    # Fetch MR details
    project = gl.projects.get(project_id)
    mr = project.mergerequests.get(mr_iid)
    title = mr.title
    description = mr.description or ""

    # Fetch diff
    diffs = mr.diffs.list()
    full_diff = ""
    for diff in diffs:
        diff_data = mr.diffs.get(diff.id)
        for change in diff_data.diffs:
            path = change['new_path']
            full_diff += f"File: {path}\n"
            full_diff += change.get('diff', '') + "\n\n"

    # Generate prompt
    prompt = REVIEW_PROMPT.format(title=title, description=description, diff=full_diff)

    # Call Gemini with JSON mode
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json"
            )
        )
    except Exception as e:
        print(f"Gemini API error: {str(e)}")
        return {
            "summary": f"Error calling Gemini API: {str(e)}",
            "recommendation": "needs fixes",
            "general_comments": [],
            "line_comments": []
        }, mr, project

    # Clean JSON string
    json_str = clean_json(response.text)

    try:
        review_json = json.loads(json_str)

        # Validate required fields
        if not isinstance(review_json, dict):
            raise ValueError("Response is not a JSON object")

        required_fields = ['summary', 'recommendation', 'general_comments', 'line_comments']
        for field in required_fields:
            if field not in review_json:
                review_json[field] = [] if field.endswith('comments') else ""

    except (json.JSONDecodeError, ValueError) as e:
        print(f"JSON parse error: {str(e)}")
        print(f"Raw response (first 500 chars): {response.text[:500]}")
        print(f"Cleaned JSON (first 500 chars): {json_str[:500]}")

        # Try to extract at least the summary
        summary_match = re.search(r'"summary":\s*"([^"]*)"', response.text)
        summary = summary_match.group(1) if summary_match else "Error parsing AI response."

        review_json = {
            "summary": summary,
            "recommendation": "needs fixes",
            "general_comments": [],
            "line_comments": []
        }

    # Time check
    elapsed = time.time() - start_time
    if elapsed > 300:  # >5min
        review_json['summary'] += f"\nAnalysis took {elapsed:.2f}s (optimize for speed)."

    return review_json, mr, project


# Function to post review to GitLab
def post_review_to_gitlab(project_id, mr_iid, review_json):
    """Post review results to GitLab MR with detailed logging"""
    print(f"Starting publication to GitLab...")
    print(f"Project ID: {project_id}, MR IID: {mr_iid}")

    try:
        # Get project and MR
        project = gl.projects.get(project_id)
        print(f"✓ Project found: {project.name}")

        mr = project.mergerequests.get(mr_iid)
        print(f"✓ MR found: {mr.title}")
        print(f"  MR URL: {mr.web_url}")

        # Post general comments
        general_comments_text = "\n".join([f"- {c}" for c in review_json.get('general_comments', [])])
        general_note = f"""## 🤖 AI Code Review

**Summary:**
{review_json['summary']}

**Recommendation:** `{review_json['recommendation'].upper()}`

**General Comments:**
{general_comments_text if general_comments_text else "No general comments."}

---
*Generated by AI Code Review Assistant*
"""

        note = mr.notes.create({'body': general_note})
        print(f"✓ General comment posted (ID: {note.id})")

        # Get diff information for inline comments
        print(f"Fetching diff information...")
        diffs = mr.diffs.list()
        if not diffs:
            print("⚠ No diffs found, skipping inline comments")
            return True

        latest_diff = mr.diffs.get(diffs[0].id)
        print(f"✓ Latest diff retrieved (ID: {latest_diff.id})")

        # Post line-specific comments as inline comments
        line_comments = review_json.get('line_comments', [])
        print(f"Posting {len(line_comments)} inline comments...")

        inline_success = 0
        inline_failed = 0

        for i, lc in enumerate(line_comments, 1):
            try:
                file_path = lc['path']
                line_number = lc['line']
                comment_body = lc['comment']

                # Find the file in the diff
                file_diff = None
                for diff_file in latest_diff.diffs:
                    if diff_file['new_path'] == file_path or diff_file['old_path'] == file_path:
                        file_diff = diff_file
                        break

                if not file_diff:
                    print(f"  ⚠ File {file_path} not found in diff, posting as regular comment")
                    # Fallback to regular comment
                    comment_text = f"**📝 {file_path}:L{line_number}**\n\n{comment_body}"
                    mr.notes.create({'body': comment_text})
                    inline_failed += 1
                    continue

                # Create inline comment with position
                position_data = {
                    'base_sha': mr.diff_refs['base_sha'],
                    'start_sha': mr.diff_refs['start_sha'],
                    'head_sha': mr.diff_refs['head_sha'],
                    'position_type': 'text',
                    'new_path': file_path,
                    'new_line': line_number
                }

                # Try to post as inline comment
                try:
                    discussion = mr.discussions.create({
                        'body': f"💡 **AI Review:**\n\n{comment_body}",
                        'position': position_data
                    })
                    print(f"  ✓ Inline comment {i}/{len(line_comments)} posted at {file_path}:L{line_number}")
                    inline_success += 1
                except Exception as inline_error:
                    # If inline fails, post as regular comment with file reference
                    print(f"  ⚠ Inline comment failed, posting as regular: {str(inline_error)}")
                    comment_text = f"**📝 {file_path}:L{line_number}**\n\n{comment_body}"
                    mr.notes.create({'body': comment_text})
                    inline_failed += 1

            except Exception as e:
                print(f"  ✗ Error posting comment {i}: {str(e)}")
                inline_failed += 1

        print(f"✓ Inline comments: {inline_success} successful, {inline_failed} as regular comments")

        # Set labels based on recommendation
        try:
            labels = list(mr.labels) if hasattr(mr, 'labels') else []
            original_labels = labels.copy()

            if review_json['recommendation'] == 'merge':
                if 'AI:ready-for-merge' not in labels:
                    labels.append('AI:ready-for-merge')
                labels = [l for l in labels if l not in ['AI:needs-review', 'AI:changes-requested']]
            elif review_json['recommendation'] == 'needs fixes':
                if 'AI:changes-requested' not in labels:
                    labels.append('AI:changes-requested')
                labels = [l for l in labels if l not in ['AI:ready-for-merge']]
            else:
                if 'AI:needs-review' not in labels:
                    labels.append('AI:needs-review')
                labels = [l for l in labels if l not in ['AI:ready-for-merge']]

            if labels != original_labels:
                mr.labels = labels
                mr.save()
                print(f"✓ Labels updated: {labels}")
            else:
                print(f"  Labels unchanged: {labels}")
        except Exception as e:
            print(f"  ⚠ Could not update labels: {str(e)}")

        print(f"✓ Publication completed successfully!")
        print(f"  Summary: {inline_success} inline comments + 1 general comment")
        return True

    except Exception as e:
        print(f"✗ Error during publication: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


# Flask for webhook (automatic reviews)
flask_app = Flask(__name__)


@flask_app.route('/webhook', methods=['POST'])
def handle_webhook():
    """Handle GitLab webhook events for automatic code reviews"""
    print("\n" + "=" * 60)
    print("📨 Webhook received!")

    data = request.json

    # Log webhook details
    print(f"Event type: {data.get('object_kind')}")

    if data.get('object_kind') == 'merge_request':
        event = data['object_attributes']['action']
        print(f"MR Action: {event}")

        if event in ['open', 'update', 'reopen']:
            project_id = data['project']['id']
            mr_iid = data['object_attributes']['iid']
            mr_title = data['object_attributes']['title']

            print(f"Project ID: {project_id}")
            print(f"MR IID: {mr_iid}")
            print(f"MR Title: {mr_title}")
            print(f"Starting automatic review...")
            print("=" * 60 + "\n")

            try:
                # Perform AI review
                review_json, _, _ = ai_review_mr(project_id, mr_iid)

                # Post review to GitLab
                post_review_to_gitlab(project_id, mr_iid, review_json)

                print("\n" + "=" * 60)
                print("✅ Automatic review completed successfully!")
                print("=" * 60 + "\n")

                return jsonify({
                    'status': 'reviewed',
                    'project_id': project_id,
                    'mr_iid': mr_iid,
                    'recommendation': review_json['recommendation']
                }), 200

            except Exception as e:
                print(f"\n❌ Webhook error: {str(e)}")
                import traceback
                traceback.print_exc()
                print("=" * 60 + "\n")

                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500
        else:
            print(f"⚠ Action '{event}' ignored (not open/update/reopen)")
            print("=" * 60 + "\n")
    else:
        print(f"⚠ Event type '{data.get('object_kind')}' ignored (not merge_request)")
        print("=" * 60 + "\n")

    return jsonify({'status': 'ignored'}), 200


@flask_app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for webhook server"""
    return jsonify({
        'status': 'ok',
        'service': 'AI Code Review Assistant',
        'mode': 'webhook'
    }), 200


# Streamlit interface (manual reviews, usability focus)
def streamlit_app():
    st.title("AI-Code Review Assistant")
    st.markdown("""
    ### Удобный интерфейс для ручного ревью MR
    Введите детали GitLab для ревью MR. Для автоматического ревью настройте webhook.
    """)

    # Initialize session state
    if 'review_data' not in st.session_state:
        st.session_state.review_data = None
    if 'project_id' not in st.session_state:
        st.session_state.project_id = ""
    if 'mr_iid' not in st.session_state:
        st.session_state.mr_iid = ""
    if 'published' not in st.session_state:
        st.session_state.published = False
    if 'publishing' not in st.session_state:
        st.session_state.publishing = False

    project_id = st.text_input("ID или путь проекта (например, namespace/project)",
                               value=st.session_state.project_id,
                               key="project_input")
    mr_iid = st.text_input("IID Merge Request (например, 42)",
                           value=st.session_state.mr_iid,
                           key="mr_input")

    if st.button("Ревью MR", key="review_btn"):
        if not project_id or not mr_iid:
            st.error("Пожалуйста, заполните оба поля")
        else:
            with st.spinner("Анализ..."):
                try:
                    review_json, mr, project = ai_review_mr(project_id, int(mr_iid))
                    st.session_state.review_data = review_json
                    st.session_state.project_id = project_id
                    st.session_state.mr_iid = mr_iid
                    st.session_state.published = False
                    st.session_state.publishing = False
                    st.success("Ревью завершено!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ошибка: {str(e)}")
                    st.session_state.review_data = None

    # Display review results if available
    if st.session_state.review_data:
        review_json = st.session_state.review_data

        st.subheader("Сводка")
        st.write(review_json['summary'])

        st.subheader("Рекомендация")
        st.write(review_json['recommendation'].upper())

        st.subheader("Общие комментарии")
        for comment in review_json.get('general_comments', []):
            st.write(f"- {comment}")

        st.subheader("Комментарии к строкам")
        for lc in review_json.get('line_comments', []):
            st.write(f"**{lc['path']}:{lc['line']}** - {lc['comment']}")

        st.divider()

        # Publish button section
        col1, col2 = st.columns([1, 4])

        with col1:
            if st.session_state.published:
                st.success("✅ Опубликовано!")
                if st.button("🔄 Опубликовать заново", key="republish_btn"):
                    st.session_state.publishing = True
                    st.rerun()
            elif st.session_state.publishing:
                with st.spinner("Публикация..."):
                    try:
                        success = post_review_to_gitlab(
                            st.session_state.project_id,
                            int(st.session_state.mr_iid),
                            review_json
                        )
                        if success:
                            st.session_state.published = True
                            st.session_state.publishing = False
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Ошибка: {str(e)}")
                        st.session_state.publishing = False
                        st.rerun()
            else:
                if st.button("📤 Опубликовать в GitLab", key="publish_btn"):
                    st.session_state.publishing = True
                    st.rerun()

        with col2:
            if st.button("🆕 Новое ревью", key="new_review_btn"):
                st.session_state.review_data = None
                st.session_state.project_id = ""
                st.session_state.mr_iid = ""
                st.session_state.published = False
                st.session_state.publishing = False
                st.rerun()


# Run modes
if __name__ == '__main__':
    mode = os.getenv('MODE', 'streamlit')  # Установите MODE=webhook для сервера
    if mode == 'webhook':
        flask_app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
    else:
        streamlit_app()  # Запуск: streamlit run app.py