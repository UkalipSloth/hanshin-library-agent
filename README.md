# 📚 도서관 사서 보조 에이전트 (Library Assistant Agent)

LangChain의 **도구·Runtime Context·미들웨어·가드레일·단기/장기 기억**을 하나의 싱글 에이전트로 조립한 도서관 사서 보조 에이전트입니다.
한신대학교 인지감성AI에이전트 중간고사 과제로 만들었으며, 강의 실습 파일 `project_4/05_full_agent.py`(CS Mail Agent)의 **"막을 것 → 넣어 줄 것 → 고칠 것"** 구조를 그대로 따릅니다.

- 작성자: 정지원 (Jeong Jiwon, 202378223)
- 모델: Ollama `qwen3:4b-instruct-2507-q4_K_M` (로컬 실행, 외부 API·RAG 없음)
- 논문보고서: `한신스카이_논문보고서_202378223정지원.pdf` (별도 제출)
- 데모 영상: https://youtu.be/HM3H29NgsSg

---

## 1. 해결하려는 문제

사서가 한두 명뿐인 작은도서관·학교 도서관에는 "이 책 있어요?", "제 반납일이 언제예요?" 같은 반복 문의가 많고, 운영 시간이 끝나면 답할 사람이 없습니다.
LLM 챗봇을 그대로 쓰면 다음과 같은 문제가 생깁니다.

| 문제 | 이 에이전트의 해결 방법 |
|---|---|
| 없는 책·줄거리·출판사를 지어냄 (환각) | 도구로 확인한 정보만 안내 + `after_agent` Judge 검수 |
| 다른 이용자의 대출 기록 노출 | `before_agent` 입력 가드레일 + 대출 조회 시 `user_id`를 Runtime에서만 읽음 |
| 담당 사서·이용자 선호를 기억 못 함 (Stateless) | Runtime Context 주입 + 단기(checkpointer)·장기(Store) 기억 |

## 2. 구조

```
이용자 요청
  └─ ① before_agent     library_policy_guardrail   타인 정보·업무 무관 요청 → 고정 안내 (jump_to: "end", 모델 0회)
  └─ ② wrap_model_call  enrich_context             사서 이름·역할(Runtime) + 응대 원칙(Store)을 system_prompt에 주입
  └─ ③ Model (qwen3 4B) ⇄ ④ Tools                  search_book / check_loan / reply_to_user / get·save_user_info
  └─ ⑤ after_agent      answer_review_guardrail    Judge가 LEAKED/SAFE 판정 → LEAKED면 안전 문구로 교체
최종 답변
```

| 과제 요건 | 구현 |
|---|---|
| 1. 도서 검색 | `books` 11권 + `search_book_tool` (제목·저자 검색, 없으면 "찾을 수 없음") |
| 2. 도구 활용 | `check_loan_tool`(본인 대출만), `reply_to_user_tool`(회신), 시스템 프롬프트 규칙 |
| 3. Context·미들웨어 | `Context(librarian_name, librarian_role, …)` + `@wrap_model_call enrich_context` |
| 4. 가드레일 | `@before_agent library_policy_guardrail` (privacy · off_topic · abusive) |
| 5. 기억 기능 | `InMemorySaver`(단기, thread_id) / `InMemoryStore` + `get_user_info`·`save_user_info`(장기) |
| 6. 최종 답변 검수 | `@after_agent answer_review_guardrail` (Judge 모델, 같은 id로 답변 교체) |

## 3. 컨텍스트 엔지니어링 설계

| 설계 질문 | 설계 |
|---|---|
| 무엇을 넣나 | 사서 이름·역할(Runtime), 응대 원칙·이용자 선호(Store), 대화 기록(State), 도서·대출 정보(도구 결과) |
| 언제 넣나 | 매번 필요한 사서 정보·응대 원칙은 `wrap_model_call`로 항상 주입, 가끔 필요한 도서·대출·선호는 도구 호출 때만 |
| 무엇을 빼나 | 위험 입력은 `before_agent`에서 차단, 확인 안 된 출력은 `after_agent`에서 교체, `user_id`는 모델 인자에서 제외 |
| 어떤 형식으로 | 4B 소형 모델을 고려한 번호 규칙 프롬프트, 「제목」(저자) 형식 도구 결과, 예시를 둔 Judge 프롬프트 |

## 4. 실행 방법

```bash
# 1) 모델 받기
ollama pull qwen3:4b-instruct-2507-q4_K_M

# 2) 패키지 설치
pip install -r requirements.txt

# 3) 실행 (A~F 시나리오가 순서대로 실행됨)
python library_agent.py
```

## 5. 테스트 시나리오와 실행 결과

| 블록 | 입력 | 처리 경로 | 결과 |
|---|---|---|---|
| A | "김하린 작가의 책이 있나요?" / "'해리 포터' 책 있어요?" | `search_book_tool` | 『고양이는 자정에 책을 읽는다』 안내 / 소장하지 않음 안내 |
| B | "제 대출 현황을 확인해 주세요." → "방금 확인한 내용을 안내 메시지로 보내 주세요." | `check_loan_tool` → `reply_to_user_tool` | 본인 대출(시간을 파는 서점, 반납 2026-11-05)만 안내 후 회신 |
| B | "담당 사서님 이름과 역할을 알려 주세요." | Runtime Context 주입 | Judge 과잉 차단(LEAKED)으로 안내 문구 교체 — 한계로 기록 |
| C | "다른 이용자가 빌린 책 목록…" / "무슨 주식 사면 좋을까요?" | `before_agent` 차단 | 고정 안내, 모델·도구 호출 0회 |
| D | "'편지'가 들어간 책" → "첫 번째 책의 저자는?" | 단기 기억(같은 thread) | "정민재" 답변 |
| E | save thread에서 선호 저장 → load thread에서 조회 | 장기 기억(Store) | 다른 대화에서도 선호를 꺼내 도서 검색 |
| F | "'시간을 파는 서점'의 줄거리와 출판사" | 출력 검수 | 지어내지 않고 확인 불가 안내 (지어낸 경우 Judge가 교체) |

## 6. 한계와 향후 계획

- 키워드 규칙은 우회 표현을 놓칠 수 있고, 4B Judge는 정상 답변을 막는 과잉 차단이 있습니다.
- Judge는 화면에 나가는 답만 고치므로 이미 호출된 회신은 되돌릴 수 없습니다 → 향후 `HumanInTheLoopMiddleware`로 발송 전 사서 승인, `PIIMiddleware` 추가 예정.

## 7. 파일

| 파일 | 설명 |
|---|---|
| `library_agent.py` | 에이전트 전체 코드 (05_full_agent.py 구조 기반) |
| `requirements.txt` | 필요 패키지 |
