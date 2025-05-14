import json
import asyncio
import time
from typing import TypedDict, List, Optional, Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_core.globals import set_debug
from pydantic import BaseModel, Field
from uuid import uuid4
from langgraph.prebuilt import ToolNode
import inspect
# --- MCP 설정 관련 ---
mcp_config_json = "mcp_config.json"

def load_mcp_config():
    """./mcp_config.json 파일에서 MCP 설정을 로드합니다."""
    try:
        with open(f"./{mcp_config_json}", "r") as f:
            return json.load(f)
    except Exception as e:
        # 오류 발생 시 메시지를 출력하고 None을 반환합니다.
        print(f"MCP 설정 파일 로드 중 오류 발생: {str(e)}")
        return None

def create_server_config():
    """로드된 MCP 설정에서 서버 구성 딕셔너리를 생성합니다."""
    config = load_mcp_config()
    server_config = {}

    # 설정 파일이 로드되었고 "mcpServers" 키가 있는지 확인합니다.
    if config and "mcpServers" in config:
        for server_name, server_config_data in config["mcpServers"].items():
            # 설정 데이터에 "command"가 있으면 stdio 전송 방식을 사용합니다.
            if "command" in server_config_data:
                server_config[server_name] = {
                    "command": server_config_data.get("command"),
                    "args": server_config_data.get("args", []), # args가 없으면 빈 리스트 사용
                    "transport": "stdio",
                }
            # 설정 데이터에 "url"이 있으면 sse 전송 방식을 사용합니다.
            elif "url" in server_config_data:
                 server_config[server_name] = {
                     "url": server_config_data.get("url"),
                     "transport": "sse",
                 }
            # 다른 방식이 있다면 여기에 추가할 수 있습니다.

    return server_config

# --- 상태 정의 ---
class AgentState(TypedDict):
    user_input: str
    initial_plan: Optional[List[str]]
    objects_to_detect: Optional[List[str]]
    unsupported_steps: Optional[List[str]]
    messages: List[AIMessage]
    location_of_object: Optional[Dict[str, List[float]]]
    location_of_current_tcp: Optional[List[float]]
    execution_sequence: List[str]

# --- 모델 정의 ---
class ClassificationOutput(BaseModel):
    is_complex: bool = Field(
        description="True if the task requires specialized processing or unsupported actions."
    )
    unsupported_steps: List[str] = Field(
        default_factory=list,
        description="List of action phrases from the user's request that cannot be executed with move, grip, or release."
    )

# Planner 에이전트 출력 형식 정의
class PlanOutput(BaseModel):
    steps_planner: List[str]
    objects_to_detect: List[str] = Field(
        description="List of object names to be detected separately from the plan steps"
    )

# --- 에이전트 함수 ---
def task_classifier_agent(llm, state: AgentState, config=None) -> Dict[str, Any]:
    """
    사용자 요청을 분석하여:
      - 단순(move, grip, release) 작업인지, 복잡 작업인지 판단
      - 복잡 작업의 경우 실행 불가 단계(unsupported_steps)를 추출 및 상태에 저장
    결과를 JSON으로 반환: is_complex, unsupported_steps, messages
    """
    system_prompt = ("""
You are a TaskClassifierAgent. Your job is to decide whether a user-provided command consists only of simple, supported actions or includes any complex/unsupported actions.

1. **Supported basic actions** (with synonyms):
   - **move**: go, navigate  
   - **grip**: grab, pick, seize, hold  
   - **release**: drop, put, place  

2. **Normalization step**  
   - Treat input **case-insensitively**.  
   - Split the input command into individual action phrases by “and”, commas, or semicolons.  
   - Convert all synonyms in each phrase to their canonical verbs:
     e.g. “Grab black box” → “grip black box”, “put white box” → “release white box”.

3. **Classification step**  
   - After normalization, if **all** phrases begin with exactly “move”, “grip” or “release”, then:
     ```json
     {"is_complex": false, "unsupported_steps": []}
     ```
   - Otherwise:
     - Set `"is_complex": true`
     - For each phrase whose verb is **not** one of the three supported verbs, extract and list the **original** verb+object text (정규화 전 문자열) under `"unsupported_steps"`.
     ```json
     {"is_complex": true, "unsupported_steps": ["fly to rooftop", "paint fence"]}
     ```

4. **Output format**  
   - **Output exactly** this JSON (no extra text):
     ```json
     {"is_complex": <true|false>, "unsupported_steps": [<string>, ...]}
     ```

"""
)


    structured = llm.with_structured_output(ClassificationOutput)
    msgs = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state['user_input'])
    ]
    out = structured.invoke(msgs, config=config)
    # 상태에 unsupported_steps 저장
    state['unsupported_steps'] = out.unsupported_steps
    # 메시지 생성
    msg = AIMessage(
        content=(
            f"Classified as {'complex' if out.is_complex else 'simple'}; "
            f"Unsupported steps: {out.unsupported_steps}"
        ),
        name="task_classifier_agent"
    )
    # 반환값
    return {
        "is_complex": out.is_complex,
        "unsupported_steps": out.unsupported_steps,
        "messages": [msg]
    }




def initial_planner_agent(llm, state: AgentState, config=None) -> Dict[str, Any]:
    # system_prompt = (
    #     "You are an expert planner AI. Based on the user's request, generate a concise abstract plan excluding detection steps, and extract all object names and surfaces to detect.\n"
    #     "1. Identify every noun phrase (objects and target surfaces) in the input and list them under `objects_to_detect` in order(But Do not include numeric coordinates (numbers with commas) as objects.).\n"
    #     "2. Break the request into logical abstract action steps—only move, grip, and release steps—omitting any detect steps.\n"
    #     "3. Preserve logical order of operations.\n"
    #     "4. Respond with exactly this JSON (no extra text): {\"steps_planner\": [...], \"objects_to_detect\": [...]}"
    # )
    system_prompt = (
    "You are an expert planner AI. Based on the user's request, generate a concise abstract plan excluding detection steps, and extract all object names and surfaces to detect.\n"
    "1. Identify every noun phrase (objects and target surfaces) in the input and list them under `objects_to_detect` in order (but do not include numeric coordinates).\n"
    "2. Break the request into logical abstract action steps—only move, grip, and release steps—omitting any detect steps.\n"
    "   * **Whenever** the user says “release X on Y”, emit **two** separate steps: first “move to Y”, then “release X”.\n"
    "3. Preserve logical order of operations.\n"
    "4. Respond with exactly this JSON (no extra text): {\"steps_planner\": [...], \"objects_to_detect\": [...]}\""
)


    # 구조화된 출력 설정
    structured_llm = llm.with_structured_output(PlanOutput)

    # LLM 호출 메시지 구성
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state['user_input'])
    ]

    # 응답 생성
    response = structured_llm.invoke(messages, config=config)

    # 상태 업데이트
    return {
        "initial_plan": response.steps_planner,
        "objects_to_detect": response.objects_to_detect,
        "messages": [AIMessage(content=f"initial_plan: {response.steps_planner}, objects to detect: {response.objects_to_detect}", name="initial_planner_agent")]
    }

async def detector_agent_without_llm(state: AgentState, tool_node, config=None):
    object_names = state.get("objects_to_detect", [])
    tcp_msg = AIMessage(content="", tool_calls=[{
        "name": "robot_get_current_tcp_pose_meters",
        "args": {},
        "id": str(uuid4()),
        "type": "tool_call"
    }])
    tcp_res = await tool_node.ainvoke({"messages": [tcp_msg]})
    try:
        tcp_pose = json.loads(tcp_res["messages"][0].content)
        state["location_of_current_tcp"] = tcp_pose
    except Exception as e:
        print(f"TCP pose error: {e}")
        state["location_of_current_tcp"] = [0.0,0.3,0.3] # 기본값으로 가정

    location_dict = {}
    for name in object_names:
        detect_msg = AIMessage(content="", tool_calls=[{
            "name": "find_object_3d_properties",
            "args": {"object_name": name},
            "id": str(uuid4()),
            "type": "tool_call"
        }])
        detect_res = await tool_node.ainvoke({"messages": [detect_msg]})
        try:
            props = json.loads(detect_res["messages"][0].content)
            location_dict[props["object_name"]] = props["position"]
        except Exception as e:
            print(f"Detection error for {name}: {e}")

    state["location_of_object"] = location_dict
    return {
        "location_of_current_tcp": state["location_of_current_tcp"],
        "location_of_object": state["location_of_object"]
    }

async def execution_sequence_planner_agent(llm, state: AgentState, config=None) -> Dict[str, Any]:
    """
    Detector 이후: initial_plan, 객체 위치 정보를 기반으로
    execution_sequence를 생성하고, 검증하고, 필요시 수정합니다.
    """

    def generate_simple_execution_sequence(state: AgentState) -> List[str]:
        """
        Detector 이후: initial_plan과 object 위치 정보를 기반으로 execution_sequence를 생성합니다.

        - move to: 물체 미보유 시 안전고도 접근, 물체 보유 시 travel_z 유지
        - grip: gripper on 직후 index 저장, 지정된 고도(travel_z)로 상승 (디버그 출력 포함)
        - release: grip 직후 생성된 move 명령 모두 제거, grip 위치에서 travel_z로 이동, target_h + held_h 위치로 이동 후 gripper off (디버그 출력 포함)
        """
        object_locations = state["location_of_object"]
        initial_plan = state["initial_plan"]

        execution_sequence: List[str] = []
        held_h = 0.0                  # 들고 있는 물체 높이
        last_move_target = None       # 마지막 'move to' 대상
        last_grip_index = 0           # grip 직후 시퀀스 길이 저장
        last_grip_pos = [0.0, 0.0]    # grip 시점 x,y 위치
        clearance = 0.03              # 안전 여유 높이
        travel_z = 0.2                # 이동 시 유지할 고도

        for step in initial_plan:
            if step.startswith("move to"):
                target = step.split("move to ", 1)[1].strip()
                if target not in object_locations:
                    coords = [float(v) for v in target.split(",")]
                    print(f"[DEBUG] move to coordinates: {coords}")
                    execution_sequence.append(f"move {coords}")
                else:
                    x, y, h = object_locations[target]
                    last_move_target = target
                    print(f"[DEBUG] move to object: {target}, x:{x}, y:{y}, h:{h}, held_h:{held_h}")
                    if held_h > 0:
                        execution_sequence.append(f"move [{x}, {y}, {travel_z}]")
                    else:
                        execution_sequence.append(f"move [{x}, {y}, {h + clearance}]")
                        execution_sequence.append(f"move [{x}, {y}, {h}]")

            elif step.startswith("grip"):
                target = step.split("grip ", 1)[1].strip()
                # 물체 높이 저장 및 디버그 출력
                x, y, h = object_locations[target]
                print(f"[DEBUG] grip -> target:{target}, x:{x}, y:{y}, h:{h}")
                held_h = h
                # 그리퍼 ON
                execution_sequence.append("gripper on")
                # grip 직후 index 저장 (gripper on 포함)
                last_grip_index = len(execution_sequence)
                # 잡은 후 travel_z로 상승
                execution_sequence.append(f"move [{x}, {y}, {travel_z}]")
                last_grip_pos = [x, y]

            elif step.startswith("release"):
                # 디버그 출력: release 시작 정보
                print(f"[DEBUG] release start -> last_move_target:{last_move_target}, last_grip_pos:{last_grip_pos}, held_h:{held_h}")
                # grip 이후 생성된 move 명령 제거
                execution_sequence = execution_sequence[:last_grip_index]
                if not last_move_target:
                    raise ValueError("Release 대상이 설정되지 않았습니다.")
                gx, gy = last_grip_pos
                print(f"[DEBUG] release -> grip 위치에서 travel_z로 이동 ({gx}, {gy}, {travel_z})")
                execution_sequence.append(f"move [{gx}, {gy}, {travel_z}]")
                # release 위치 계산 및 디버그 출력
                x, y, tgt_h = object_locations[last_move_target]
                print(f"[DEBUG] release -> target:{last_move_target}, x:{x}, y:{y}, target_h:{tgt_h}")
                release_z = tgt_h + held_h + 0.02
                print(f"[DEBUG] release_z 계산 -> release_z:{release_z}, target_h:{tgt_h}, held_h:{held_h}")
                execution_sequence.append(f"move [{x}, {y}, {release_z}]")
                execution_sequence.append("gripper off")
                print(f"execution_sequence of release gripper: {execution_sequence}")
                held_h = 0.0

            else:
                print(f"[DEBUG] 기타 명령: {step}")
                execution_sequence.append(step)
        print(f"[DEBUG] 최종 execution_sequence: {execution_sequence}")

        return execution_sequence





    async def validate_execution_sequence(llm, state, generated_sequence, config=None):
        system_prompt = """
You are a robot motion plan validator.

Inputs:
- initial_plan: List of high-level steps like "move to object", "grip object", etc.
- object_locations: Map of object names to coordinates.
- execution_sequence: List of low-level commands like "move [x,y,z]", "gripper on", "gripper off".

Rules:
- When moving to an object:
  - First move to 5cm above the object's position (z + 0.05).
- When gripping:
  - After reaching 5cm above, move vertically to the object's z coordinate.
- After releasing:
  - Immediately move 5cm upwards (z + 0.05).

Instructions:
- ONLY check whether execution_sequence strictly follows the rules.
- DO NOT regenerate plans.
- DO NOT infer missing steps.
- Output only:
  - "pass" if all rules are followed correctly
  - or "fail" with explanation of the first violation found.
"""
        context = {
            "initial_plan": state["initial_plan"],
            "object_locations": state["location_of_object"],
            "execution_sequence": generated_sequence
        }

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)}
        ]

        response = await llm.ainvoke(messages, config=config)
        return response

    async def fix_execution_sequence(llm, state, generated_sequence, config=None):
        system_prompt = """
        You are a robot motion plan fixer.

        Task:
        - Given an initial_plan, object_locations, and an invalid execution_sequence, 
        - Correct the execution_sequence so that it follows the rules exactly.

        Rules:
        - Move to 5cm above the target before approaching.
        - After reaching 5cm above, move vertically down to the object's Z for gripping.
        - After releasing, ascend 5cm vertically.

        Instructions:
        - Output ONLY a corrected execution_sequence list.
        - NO explanations, NO comments.
        """
        context = {
            "initial_plan": state["initial_plan"],
            "object_locations": state["location_of_object"],
            "invalid_execution_sequence": generated_sequence
        }

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)}
        ]

        response = await llm.ainvoke(messages, config=config)
        return response

    # --- 본체 실행 ---

    if state["initial_plan"] is None or state["location_of_object"] is None or state["location_of_current_tcp"] is None:
        raise ValueError("Initial plan, object locations, and current TCP location must be available before execution planning.")

    generated_sequence = generate_simple_execution_sequence(state)

    # validation_result = await validate_execution_sequence(llm, state, generated_sequence, config=config)

    # if 'pass' in validation_result.content.lower():
    #     final_sequence = generated_sequence
    #     validation_status = "pass"
    # else:
    #     fix_result = await fix_execution_sequence(llm, state, generated_sequence, config=config)
    #     try:
    #         fixed_sequence = json.loads(fix_result.content)
    #         final_sequence = fixed_sequence
    #         validation_status = "fixed"
    #     except Exception as e:
    #         raise ValueError(f"Failed to fix execution sequence: {e}")

    # 최종 저장
    state["execution_sequence"] = generated_sequence

    msg = AIMessage(
        content=f"Execution sequence planning completed. Validation: {generated_sequence}",
        name="execution_sequence_planner_agent"
    )

    return {
        "execution_sequence": generated_sequence,
        "messages": [msg]
    }

async def call_tool_and_wait(tool_node, tool_call, config=None):
    response = await tool_node.ainvoke({"messages": [tool_call]}, config=config)
    # print(f"Tool response: {response}")

    if not response or not response.get("messages"):
        raise RuntimeError("No response received from tool.")

    # ToolMessage 내용 체크
    tool_msg = response["messages"][0]
    if hasattr(tool_msg, 'status') and tool_msg.status == 'error':
        raise RuntimeError(f"Tool execution failed: {tool_msg.content}")

    return response


async def execution_runner_agent(state: AgentState, tool_node, config=None) -> Dict[str, Any]:
    """
    execution_sequence를 읽어서 실제 로봇 명령을 tool_node로 실행합니다.
    """

    if not state.get("execution_sequence"):
        raise ValueError("No execution sequence to run.")

    for command in state["execution_sequence"]:
        command = command.strip()

        if command.startswith("move"):
            # move [x, y, z] 파싱
            coords_text = command[len("move"):].strip()
            coords = json.loads(coords_text.replace("'", '"'))  # 안전하게 리스트로 변환

            move_msg = AIMessage(content="", tool_calls=[{
                "name": "robot_execute_trajectorys",
                "args": {"a_list": [coords]},
                "id": str(uuid4()),
                "type": "tool_call"
            }])

            print(f"Executing move to {coords}")
            response = await call_tool_and_wait(tool_node, move_msg, config=config)
            # print(f"Move response: {response}")
        elif command == "gripper on":
            grip_on_msg = AIMessage(content="", tool_calls=[{
                "name": "robot_turn_on_suction_gripper",
                "args": {},
                "id": str(uuid4()),
                "type": "tool_call"
            }])

            print("Turning gripper ON")
            await call_tool_and_wait(tool_node, grip_on_msg, config=config)

            # --- 추가: 물체 붙을 때까지 수직으로 내리기 ---
            attached = False
            current_tcp = None  # 현재 tcp 위치 기억

            for try_idx in range(20):  # 최대 20번까지 (0.01m씩 내려가면 20cm)
                # 그리퍼 붙었는지 확인
                check_msg = AIMessage(content="", tool_calls=[{
                    "name": "robot_check_attached_gripper",
                    "args": {},
                    "id": str(uuid4()),
                    "type": "tool_call"
                }])

                check_response = await call_tool_and_wait(tool_node, check_msg, config=config)
                if check_response and "Object is attached by suction." in str(check_response):
                    print("Object attached successfully!")
                    attached = True
                    break

                # 아직 안붙었으면 조금 더 내려가기
                if current_tcp is None:
                    # 현재 TCP 처음만 받아오기
                    tcp_pose_msg = AIMessage(content="", tool_calls=[{
                        "name": "robot_get_current_tcp_pose_meters",
                        "args": {},
                        "id": str(uuid4()),
                        "type": "tool_call"
                    }])
                    tcp_response = await call_tool_and_wait(tool_node, tcp_pose_msg, config=config)
                    current_tcp = json.loads(tcp_response["messages"][0].content)
                    current_tcp = [float(x) for x in current_tcp]

                # z 축을 0.01m 내리기
                current_tcp[2] -= 0.005
                move_down_msg = AIMessage(content="", tool_calls=[{
                    "name": "robot_execute_trajectorys",
                    "args": {"a_list": [current_tcp]},
                    "id": str(uuid4()),
                    "type": "tool_call"
                }])

                print(f"Descending to {current_tcp}")
                await call_tool_and_wait(tool_node, move_down_msg, config=config)

            if not attached:
                raise RuntimeError("Failed to attach object after descending.")
        elif command == "gripper off":
            grip_off_msg = AIMessage(content="", tool_calls=[{
                "name": "robot_turn_off_suction_gripper",
                "args": {},
                "id": str(uuid4()),
                "type": "tool_call"
            }])

            print("Turning gripper OFF")
            response = await call_tool_and_wait(tool_node, grip_off_msg, config=config)
            # print(f"Gripper OFF response: {response}")
        else:
            print(f"Unknown command: {command}")
            continue

    # 모든 명령 완료 후
    done_msg = AIMessage(content="Execution sequence completed successfully.", name="execution_runner_agent")

    return {
        "messages": [done_msg]
    }


# --- Graph helper ---


def wrap_async(fn, **fixed_kwargs):
    """
    어떤 함수든 (llm, state, config) 이든 (state, config, tool_node) 이든
    알아서 state, config를 넣고 나머지는 fixed_kwargs로 넘긴다.
    """
    sig = inspect.signature(fn)

    async def wrapper(state, config):
        # 함수 인자 이름 가져오기
        param_names = list(sig.parameters.keys())

        # 준비할 인자 리스트
        args = []
        kwargs = {}

        for name in param_names:
            if name == "state":
                args.append(state)
            elif name == "config":
                kwargs["config"] = config
            elif name in fixed_kwargs:
                args.append(fixed_kwargs[name])
            else:
                # 없는 인자는 기본값을 쓰게 냅둬
                pass

        return await fn(*args, **kwargs)

    return wrapper


# --- Graph 생성 ---
def create_total_graph(llm, tools):
    graph = StateGraph(AgentState)
    tool_node = ToolNode(tools)

    graph.add_node("TaskClassifierAgent", lambda state, config: task_classifier_agent(llm, state, config))
    graph.add_node("InitialPlannerAgent",lambda state, config: initial_planner_agent(llm, state, config))
    graph.add_node("DetectorAgent", wrap_async(detector_agent_without_llm, tool_node=tool_node))
    graph.add_node("ExecutionPlannerAgent", wrap_async(execution_sequence_planner_agent, llm=llm))
    graph.add_node("ExecutionRunnerAgent", wrap_async(execution_runner_agent, tool_node=tool_node))


    def route_based_on_classification(state: AgentState) -> str:
        if state.get("unsupported_steps"):
            return "complex"
        return "simple"

    graph.add_conditional_edges(
        "TaskClassifierAgent",
        route_based_on_classification,
        {
            "complex": END,
            "simple": "InitialPlannerAgent"
        }
    )

    graph.add_edge("InitialPlannerAgent", "DetectorAgent")
    graph.add_edge("DetectorAgent", "ExecutionPlannerAgent")
    graph.add_edge("ExecutionPlannerAgent", "ExecutionRunnerAgent")
    graph.add_edge("ExecutionRunnerAgent", END)


    graph.set_entry_point("TaskClassifierAgent")
    return graph.compile()

# --- Main ---
async def main():
    llm = ChatOllama(model="qwen3:14b", temperature=0)
    mcp_server_configs = create_server_config()
    print("--- MCP Client Initializing ---")
    async with MultiServerMCPClient(mcp_server_configs) as client:
        print("--- MCP Client Initialized ---")
        tools = client.get_tools()
        compiled_graph = create_total_graph(llm, tools)

        query = input("Enter your command: ")
        initial_state: AgentState = {
            "user_input": query,
            "initial_plan": None,
            "objects_to_detect": None,
            "unsupported_steps": None,
            "messages": [],
            "location_of_object": None,
            "location_of_current_tcp": None
        }

        config = RunnableConfig(recursion_limit=150)

        async for event in compiled_graph.astream(initial_state, config=config, stream_mode="updates"):
            for node, output in event.items():
                print(f"[{node}] output: {output}")

if __name__ == "__main__":
    try: asyncio.get_event_loop().run_until_complete(main())
    except KeyboardInterrupt: print("\nInterrupted.")
    except Exception as e: print(f"\nCritical error: {e}"); import traceback; traceback.print_exc()
