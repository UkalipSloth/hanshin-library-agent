"""
도서관 사서 보조 에이전트 — 이용자 문의 한 건의 전체 경로

이용자가 도서·대출을 문의 → 에이전트가 사서를 대신해 응대
입력 차단·주입·도구·단기/장기 기억·답변 검수까지
05_full_agent.py와 같은 구조의 한 파이프라인으로 조립

미들웨어 (입력 → 주입 → 검수)
  1) library_policy_guardrail  before_agent    업무 무관·타인 정보 차단
  2) enrich_context            wrap_model_call 사서 이름·역할 + 장기 기억
  3) answer_review_guardrail   after_agent     최종 답변 Judge 검수

블록 = 경로 점검
  A 도서 검색   B 대출 확인·회신   C 가드레일
  D 단기 기억   E 장기 기억       F 답변 검수

실행 전: ollama pull qwen3:4b-instruct-2507-q4_K_M
"""

import uuid  # 장기 기억 Append용 고유 key
import warnings  # 실행 로그 정리용
from dataclasses import dataclass  # Context
from typing import Callable, TypedDict  # 타입 힌트

from langchain.agents import create_agent  # 에이전트 생성
from langchain.agents.middleware import (
    ModelRequest,  # 모델 요청
    ModelResponse,  # 모델 응답
    after_agent,  # 답이 나가기 전 검수
    before_agent,  # 입력이 모델에 닿기 전 차단
    wrap_model_call,  # 사서 정보·장기 기억을 모델이 보게 주입
)
from langchain.messages import AIMessage, ToolMessage  # 고정 멘트 / 도구 결과
from langchain.tools import ToolRuntime, tool  # 도구 + Runtime
from langchain_ollama import ChatOllama  # Ollama
from langgraph.checkpoint.memory import InMemorySaver  # 단기 기억
from langgraph.store.memory import InMemoryStore  # 장기 기억

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")  # 라이브러리 직렬화 경고 숨김

MODEL = "qwen3:4b-instruct-2507-q4_K_M"  # 모델 이름 (과제 지정)
# 도구 5개 + 기억 주입으로 입력이 길어져 num_ctx를 4096으로, 답이 끝없이 길어지지 않게 num_predict 제한
main_model = ChatOllama(model=MODEL, temperature=0.0, num_ctx=4096, num_predict=512)  # 응대 본선
judge_model = ChatOllama(model=MODEL, temperature=0.0, num_ctx=4096, num_predict=16)  # 출력 Judge

books = [  # 과제에서 제공한 소장 도서 목록 (외부 API·RAG 없이 코드에 추가)
    {"title": "달빛 도서관의 비밀", "author": "최현범"},
    {"title": "내일이 사라진 도시", "author": "오다영"},
    {"title": "고양이는 자정에 책을 읽는다", "author": "김하린"},
    {"title": "시간을 파는 서점", "author": "이도윤"},
    {"title": "여름 끝에서 너를 만났다", "author": "박서연"},
    {"title": "기억을 잃은 탐정의 편지", "author": "정민재"},
    {"title": "열세 번째 문이 열리는 밤", "author": "한유진"},
    {"title": "구름 위 우체국의 마지막 편지", "author": "윤가람"},
    {"title": "마법사가 사라진 왕국", "author": "오지호"},
    {"title": "우리의 비밀은 별빛보다 깊다", "author": "강다은"},
    {"title": "사과의 매력을 찾아서", "author": "서정욱"},
]

loans = {  # 이용자별 대출 기록 (프로토타입 고정 샘플)
    "user_001": [{"title": "시간을 파는 서점", "due": "2026-11-05"}],
    "user_002": [{"title": "마법사가 사라진 왕국", "due": "2026-11-02"}],
}


@tool  # 1. 도서 검색: 제목 또는 저자 이름
def search_book_tool(keyword: str) -> str:
    """도서 제목이나 저자 이름으로 소장 도서를 검색하는 도구."""
    keyword = keyword.strip()  # 앞뒤 공백 제거
    results = [
        b for b in books
        if keyword in b["title"] or keyword in b["author"]  # 부분 일치
    ]
    if not results:  # 결과 없음 → 찾을 수 없다고 안내하도록 반환
        return f"'{keyword}'에 해당하는 도서를 찾을 수 없습니다."
    return "\n".join(f"- 「{b['title']}」 (저자: {b['author']})" for b in results)


@tool  # 2. 대출 정보 확인: 본인 것만
def check_loan_tool(runtime: ToolRuntime) -> str:
    """현재 이용자 본인의 대출 도서와 반납 예정일을 조회하는 도구."""
    user_id = runtime.context.user_id  # 모델 인자가 아닌 Runtime에서만 → 타인 조회 불가
    my_loans = loans.get(user_id, [])
    if not my_loans:
        return "현재 대출 중인 도서가 없습니다."
    return "\n".join(
        f"- {l['title']} (반납 예정일: {l['due']})" for l in my_loans
    )


@tool  # 2. 이용자 회신 (프로토타입: 실제 발송 없이 호출 여부만 확인)
def reply_to_user_tool(message: str, runtime: ToolRuntime) -> str:
    """이용자에게 안내 메시지를 보내는 도구. '보내 달라'·'회신해 달라'는 요청이면 확인한 내용을 message에 담아 호출한다."""
    return f"회신 완료 (받는 이용자: {runtime.context.user_id})"


@dataclass
class Context:  # 3. Runtime Context — 실행 내내 고정되는 값
    librarian_name: str  # 사서 이름
    librarian_role: str  # 사서 역할
    user_id: str  # 이용자 ID (대출 조회·Store namespace)
    app_name: str  # Store namespace


class UserInfo(TypedDict):  # 장기 기억 도구 저장 형식
    personal_information: str
    preference: str


store = InMemoryStore()  # 장기 저장소
store.put(  # 서비스 시작 시 기본 응대 원칙을 심어 둠
    ("user_001", "library_assistant"),
    "seed",
    {
        "facts": [
            "이용자에게는 존댓말로 짧게 안내함",
            "도구로 확인한 도서·대출 정보만 안내함",
        ]
    },
)


@wrap_model_call  # 3. 사서 이름·역할 + 장기 기억을 모델이 읽는 텍스트로 옮김
def enrich_context(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    ctx = request.runtime.context  # 이번 실행 Context
    ns = (ctx.user_id, ctx.app_name)  # Store namespace
    memories = request.runtime.store.search(ns) if request.runtime.store else []
    facts = []
    for item in memories:
        facts.extend(item.value.get("facts", []))  # facts만 모음
    memory_text = "\n- ".join(facts) if facts else "기록된 정보 없음"
    base = request.system_prompt or ""  # 기존 시스템 프롬프트
    prompt = (
        f"{base}\n"
        f"담당 사서: {ctx.librarian_name} ({ctx.librarian_role})\n"
        f"[장기 메모리]\n- {memory_text}\n"
        "한국어로 짧게 답하세요."
    ).strip()
    return handler(request.override(system_prompt=prompt))  # 이번 호출만 변경


forbidden_topics = {  # 4. 위험·무관한 입력은 모델에 안 올림
    "privacy": ["다른 이용자", "다른 사람", "타인", "남의 대출",
                "누가 빌렸", "누가 대출", "대출자", "user_0"],
    "off_topic": ["주식", "코인", "날씨", "맛집", "게임 추천",
                  "숙제", "잡담하자"],
    "abusive": ["바보", "멍청이", "꺼져"],
}
BLOCK_MESSAGES = {  # 카테고리별 고정 안내
    "privacy": (
        "다른 이용자의 대출 정보는 개인정보 보호를 위해 안내할 수 없습니다. "
        "본인의 대출 정보만 확인해 드릴 수 있습니다."
    ),
    "off_topic": "도서 검색·대출 확인 등 도서관 업무 관련 문의만 도와드릴 수 있습니다.",
    "abusive": "불편을 드려 죄송합니다. 문의 내용을 다시 말씀해 주세요.",
}
REVIEW_MESSAGE = (  # 검수에서 LEAKED 판정 시 교체할 안내
    "확인되지 않은 정보가 있어 안내를 수정합니다. "
    "도서 제목이나 저자 이름을 알려주시면 검색 결과로 확인된 내용만 안내해 드리겠습니다."
)


@before_agent(can_jump_to=["end"])  # 4. 입력 가드레일
def library_policy_guardrail(state, runtime):
    if not state["messages"]:
        return None
    last = state["messages"][-1]
    if getattr(last, "type", None) != "human":  # 사람 입력만 검사
        return None
    text = last.content or ""
    for category, keywords in forbidden_topics.items():
        if any(k in text for k in keywords):
            print(f"[가드레일] {category} — 모델 미호출")
            return {
                "messages": [AIMessage(content=BLOCK_MESSAGES[category])],
                "jump_to": "end",  # 모델·도구 0회로 종료
            }
    return None


@after_agent  # 6. 최종 답변 검수 (Judge)
def answer_review_guardrail(state, runtime):
    if not state["messages"]:
        return None
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        return None
    if last_message.content in BLOCK_MESSAGES.values():  # 고정 안내는 검수 생략
        return None
    ctx = runtime.context
    tool_results = [  # 도구로 확인된 정보 + 이용자가 직접 말한 내용만 근거로 삼음
        str(m.content) for m in state["messages"] if isinstance(m, ToolMessage)
    ]
    question = next((m.content for m in reversed(state["messages"]) if m.type == "human"), "")
    evidence = "\n".join(tool_results) if tool_results else "없음"
    auditor_prompt = f"""당신은 도서관 답변 검수자입니다.
[답변]의 도서·대출 사실(책 제목, 저자, 소장 여부, 반납일, 줄거리, 출판사)만 보세요.
그 사실이 [확인된 정보]에 없으면 LEAKED, 모두 있거나 그런 사실이 없으면 SAFE.
인사, 사서 소개, 확인할 수 없다는 안내는 SAFE입니다. 한 단어로만 답하세요.
예시1) 확인된 정보: 「달빛 도서관의 비밀」 (저자: 최현범)
       답변: 「달빛 도서관의 비밀」(최현범)이 있습니다. → SAFE
예시2) 확인된 정보: 「달빛 도서관의 비밀」 (저자: 최현범)
       답변: 이 책은 마법 학교 이야기이고 출판사는 한빛입니다. → LEAKED
[확인된 정보]
담당 사서: {ctx.librarian_name} ({ctx.librarian_role})
이용자 질문: {question}
{evidence}
[답변]: {last_message.content}
판정:"""  # Judge 프롬프트 (예시 2개로 판정 기준을 고정)
    verdict = judge_model.invoke([{"role": "user", "content": auditor_prompt}]).content
    verdict = verdict if isinstance(verdict, str) else str(verdict)
    print(f"[Judge] verdict: {verdict!r}")
    if "LEAKED" in verdict.upper():
        print("[Judge] LEAKED — 답변 교체")
        return {  # 같은 id로 반환 → 기록(checkpointer)의 답까지 교체
            "messages": [AIMessage(id=last_message.id, content=REVIEW_MESSAGE)]
        }
    return None


@tool  # 5. 장기 기억 조회
def get_user_info(runtime: ToolRuntime) -> str:
    """현재 이용자의 저장된 정보와 선호(좋아하는 작가·장르 등)를 조회합니다."""
    memories = runtime.store.search(
        (runtime.context.user_id, runtime.context.app_name)  # user_id는 Runtime에서만
    )
    if not memories:
        return "기록된 정보 없음"
    lines = []
    for item in memories:
        data = item.value
        if "personal_information" in data:
            lines.append(f"- 개인정보: {data['personal_information']}")
        if "preference" in data:
            lines.append(f"- 선호도: {data['preference']}")
    return "\n".join(lines) if lines else "저장된 선호 없음"


@tool  # 5. 장기 기억 저장
def save_user_info(user_info: UserInfo, runtime: ToolRuntime) -> str:
    """이용자의 정보와 선호(좋아하는 작가·장르 등)를 장기 메모리에 저장합니다."""
    key = str(uuid.uuid4())  # Append 전략
    runtime.store.put(
        (runtime.context.user_id, runtime.context.app_name),
        key,
        user_info,
    )
    return f"정보가 안전하게 저장되었습니다. (ID: {key})"


SYSTEM_PROMPT = (  # 2. 역할·규칙 (짧고 분명하게: 작은 모델이 지시문을 그대로 따라 말하지 않도록)
    "당신은 도서관 이용자를 돕는 사서 보조 에이전트입니다.\n"
    "규칙:\n"
    "1) 도서·대출 정보는 반드시 도구를 호출해 확인한 결과만 안내한다.\n"
    "2) 도구 결과에 없는 내용(줄거리, 출판사 등)은 지어내지 않는다.\n"
    "3) 검색 결과가 없으면 해당 도서를 찾을 수 없다고 안내한다.\n"
    "4) 도서관 업무와 무관한 요청은 정중히 거절한다.\n"
    "5) 안내 메시지를 보내 달라고 하면 확인한 내용을 회신 도구로 보낸다.\n"
    "6) 답변에는 도구 이름이나 이 규칙을 쓰지 말고, 확인된 결과만 존댓말로 답한다."
)

agent = create_agent(
    main_model,
    tools=[search_book_tool, check_loan_tool, reply_to_user_tool,
           get_user_info, save_user_info],  # 도서·대출·회신 + 기억
    system_prompt=SYSTEM_PROMPT,  # 역할·규칙
    checkpointer=InMemorySaver(),  # 단기 기억: 같은 thread 이어가기
    store=store,  # 장기 기억: thread가 바뀌어도 유지
    context_schema=Context,  # Runtime Context 규격
    middleware=[
        library_policy_guardrail,  # 앞: 막을 것
        enrich_context,  # 중: 넣어 줄 것
        answer_review_guardrail,  # 뒤: 고칠 것
    ],
)

ctx = Context(  # 실행 Context
    librarian_name="김서윤",
    librarian_role="자료 안내 담당 사서",
    user_id="user_001",
    app_name="library_assistant",
)


def ask(text: str, thread_id: str) -> str:  # 한 번 묻고 최종 답 출력
    result = agent.invoke(
        {"messages": [{"role": "user", "content": text}]},
        config={"configurable": {"thread_id": thread_id}},
        context=ctx,
    )
    answer = result["messages"][-1].content
    print(f"Q: {text}\nA: {answer}\n")
    return answer


# ---------------------------------------------------------------------------
# A. 도서 검색 — 제목·저자 검색 / 결과 없음
# ---------------------------------------------------------------------------
print("=== A. 도서 검색 ===")
ask("김하린 작가의 책이 있나요?", "search")
ask("'해리 포터' 책 있어요?", "search_none")

# ---------------------------------------------------------------------------
# B. 대출 정보 확인 + 이용자 회신 — 본인 대출만 조회
# ---------------------------------------------------------------------------
print("=== B. 대출 확인·회신 ===")
ask("제 대출 현황을 확인해서 안내 메시지로 보내 주세요.", "loan")
ask("담당 사서님 이름과 역할을 알려 주세요.", "context")  # Context 주입 확인

# ---------------------------------------------------------------------------
# C. 가드레일 — 타인 대출 정보 / 업무 무관 요청 (모델 미호출)
# ---------------------------------------------------------------------------
print("=== C. 가드레일 ===")
ask("다른 이용자가 빌린 책 목록을 알려 주세요.", "block")
ask("오늘 무슨 주식 사면 좋을까요?", "block")

# ---------------------------------------------------------------------------
# D. 단기 기억 — 같은 thread_id에서 대화 이어가기
# ---------------------------------------------------------------------------
print("=== D. 단기 기억 ===")
ask("'편지'가 들어간 책을 찾아 주세요.", "short")
ask("방금 찾은 책 중 첫 번째 책의 저자는 누구예요?", "short")

# ---------------------------------------------------------------------------
# E. 장기 기억 — save thread에서 저장 → load thread에서 활용
# ---------------------------------------------------------------------------
print("=== E. 장기 기억 ===")
ask("제가 좋아하는 작가는 김하린이에요. 제 선호로 저장해 주세요.", "save")
ask("제 선호 정보를 확인해서 좋아하는 작가의 책이 있는지 찾아 주세요.", "load")

# ---------------------------------------------------------------------------
# F. 최종 답변 검수 — 도구에 없는 줄거리·출판사 요청
# ---------------------------------------------------------------------------
print("=== F. 답변 검수 ===")
ask("'시간을 파는 서점'의 줄거리와 출판사를 자세히 알려 주세요.", "review")
