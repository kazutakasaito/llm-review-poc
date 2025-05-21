"""
Multi-persona LLM reviewer - stable JP version (2025-05) - v2
--------------------------------------------------------
エンジニア / PdM / アーキテクトが JSON 形式で指摘を返し、
Supervisor がマージして review.json に出力します。
複数のストーリーファイルを処理できるように改善。
"""

import json
import pathlib
import sys
from langchain_openai import ChatOpenAI
from langgraph.graph import MessageGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# ---------- Persona settings ----------
PERSONAS = {
    "engineer": dict(
        prompt=(
            "あなたは経験15年のシニアソフトウェアエンジニアです。\n"
            "ユーザーストーリーを実装観点（依存関係・複雑度・DoR 達成状況など）でレビューしてください。\n"
            "結果を必要なだけJSON 配列 [{severity, comment, line}] だけで返してください。\n"
            "severityは 'high', 'medium', 'low' のいずれかです。\n"
            "lineは指摘箇所の行番号(整数)ですが、特定できない場合はnullにしてください。\n"
            "（コメント件数に上限を設けず、重要度順にたくさん指摘して構いません）"
        ),
        temperature=0,
    ),
    "pdm": dict(
        prompt=(
            "あなたはプロダクトマネージャーです。\n"
            "ビジネス価値と KPI への紐づきの観点でレビューしてください。\n"
            "結果を必要なだけJSON 配列 [{severity, comment, line}] だけで返してください。\n"
            "severityは 'high', 'medium', 'low' のいずれかです。\n"
            "lineは指摘箇所の行番号(整数)ですが、特定できない場合はnullにしてください。\n"
            "（コメント件数に上限を設けず、重要度順にたくさん指摘して構いません）"
        ),
        temperature=0.2,
    ),
    "architect": dict(
        prompt=(
            "あなたはソフトウェアアーキテクトです。\n"
            "非機能要件（性能・可用性・セキュリティなど）のリスクをレビューしてください。\n"
            "結果を必要なだけJSON 配列 [{severity, comment, line}] だけで返してください。\n"
            "severityは 'high', 'medium', 'low' のいずれかです。\n"
            "lineは指摘箇所の行番号(整数)ですが、特定できない場合はnullにしてください。\n"
            "（コメント件数に上限を設けず、重要度順にたくさん指摘して構いません）"
        ),
        temperature=0,
    ),
}

# ---------- LLM factory (JSON-mode ON) ----------
def make_llm(cfg):
    """LLMクライアントを生成します。JSONモードを有効にします。"""
    return ChatOpenAI(
        model="gpt-3.5-turbo",  # or "gpt-3.5-turbo" for cost saving
        temperature=cfg["temperature"],
        max_tokens=1024,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

# ---------- Graph Definition ----------
def create_workflow():
    """レビューワークフローのグラフを構築します。"""
    graph_builder = MessageGraph()

    # 各ペルソナのノードを定義
    for persona_name, config in PERSONAS.items():
        llm = make_llm(config)
        
        def persona_node(messages, persona_name_node=persona_name, persona_prompt=config["prompt"]):
            # messagesにはこれまでの会話履歴が入る。最後のメッセージが対象のストーリー。
            if not messages:
                # 通常はSTARTから呼ばれるのでここには来ない想定
                return AIMessage(role=persona_name_node, content="[]")

            # 最後のユーザーメッセージ（ストーリー本文）を取得
            # HumanMessage以外にもSystemMessageでファイル名などを渡すことも検討可能
            story_content = ""
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage) and msg.role == "user_story_content":
                    story_content = msg.content
                    break
            
            if not story_content:
                 # ストーリー内容が見つからない場合は空のJSON配列を返す
                print(f"Warning: Story content not found for persona {persona_name_node}. Returning empty list.", file=sys.stderr)
                return AIMessage(role=persona_name_node, content="[]")

            # プロンプトを組み立て
            # SystemMessageを使用して、より明確に役割と指示を与える
            full_prompt_messages = [
                SystemMessage(content=persona_prompt),
                HumanMessage(content=f"--- 対象ストーリー ---\n{story_content}\n\n### 出力は指示されたJSONオブジェクトのみ ###")
            ]
            
            try:
                response = llm.invoke(full_prompt_messages)
                # LLMからのレスポンスが空や不正なJSONでないか確認
                if not response.content or not response.content.strip().startswith("["):
                    print(f"Warning: Invalid JSON array from {persona_name_node}: {response.content}", file=sys.stderr)
                    return AIMessage(role=persona_name_node, content="[]") # 空の配列を返す
                # JSONとしてパースできるか念のため確認
                try:
                    json.loads(response.content)
                except json.JSONDecodeError:
                    print(f"Warning: Failed to parse JSON from {persona_name_node}: {response.content}", file=sys.stderr)
                    return AIMessage(role=persona_name_node, content="[]") # 空の配列を返す
                return AIMessage(role=persona_name_node, content=response.content)
            except Exception as e:
                print(f"Error invoking LLM for {persona_name_node}: {e}", file=sys.stderr)
                return AIMessage(role=persona_name_node, content="[]") # エラー時も空の配列

        graph_builder.add_node(persona_name, persona_node)
        graph_builder.add_edge(START, persona_name) # STARTから各ペルソナへ
        graph_builder.add_edge(persona_name, "merge_reviews") # 各ペルソナからマージノードへ

    # レビュー結果をマージするノード
    def merge_node(messages):
        merged_reviews = {}
        # AIMessageで、かつroleがペルソナ名であるものを収集
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.role in PERSONAS:
                try:
                    # contentがJSON配列文字列であることを期待
                    persona_review_list = json.loads(msg.content)
                    merged_reviews[msg.role] = persona_review_list
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from {msg.role}: {msg.content}", file=sys.stderr)
                    merged_reviews[msg.role] = [{"severity": "error", "comment": f"Failed to parse review from {msg.role}.", "line": None}]
                except Exception as e:
                    print(f"Error processing message from {msg.role}: {e}", file=sys.stderr)
                    merged_reviews[msg.role] = [{"severity": "error", "comment": f"Unexpected error processing review from {msg.role}.", "line": None}]


        # 最終的な出力をHumanMessageやSystemMessageではなく、AIMessage(role="supervisor") などでラップして返す
        # ここでは、直接JSON文字列を返すのではなく、次のステップで処理しやすいようにPythonのdictとして返す
        # ただし、LangGraphのMessageGraphはMessageオブジェクトを期待するため、AIMessageでラップする
        return AIMessage(role="supervisor", content=json.dumps(merged_reviews, ensure_ascii=False))

    graph_builder.add_node("merge_reviews", merge_node)
    # 'merge_reviews' の後に END を設定
    graph_builder.add_edge("merge_reviews", END)
    
    return graph_builder.compile()

# ---------- CLI ----------
def main():
    if len(sys.argv) < 2:
        print("Usage: python reviewer_agents.py <story1.md> [story2.md ...]", file=sys.stderr)
        sys.exit(1)

    story_files = sys.argv[1:]
    all_results = []
    
    workflow = create_workflow()

    for story_file_path_str in story_files:
        story_path = pathlib.Path(story_file_path_str)
        if not story_path.is_file():
            print(f"Error: File not found - {story_file_path_str}", file=sys.stderr)
            all_results.append({
                "file_path": story_file_path_str,
                "error": "File not found",
                "review": {}
            })
            continue

        try:
            story_text = story_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Error: Could not read file {story_file_path_str}: {e}", file=sys.stderr)
            all_results.append({
                "file_path": story_file_path_str,
                "error": f"Could not read file: {e}",
                "review": {}
            })
            continue
        
        # ワークフローに渡す初期メッセージ。HumanMessageのroleを工夫してファイル内容であることを示す
        # LangGraphでは通常、リスト形式でメッセージを渡す
        initial_message = [HumanMessage(content=story_text, role="user_story_content")]
        
        print(f"Reviewing {story_file_path_str}...", file=sys.stderr)
        
        # ワークフローを実行
        # invokeの戻り値は、グラフ内の各ノードが返したMessageオブジェクトのリスト
        final_state_messages = workflow.invoke(initial_message)
        
        # 最後のメッセージ（merge_reviewsノードからの出力）を取得
        merged_review_content = "{}" # デフォルトは空のJSONオブジェクト文字列
        for msg in reversed(final_state_messages):
            if isinstance(msg, AIMessage) and msg.role == "supervisor":
                merged_review_content = msg.content
                break
        
        try:
            review_data = json.loads(merged_review_content)
            all_results.append({
                "file_path": story_file_path_str,
                "review": review_data
            })
        except json.JSONDecodeError as e:
            print(f"Error: Could not decode final merged review JSON for {story_file_path_str}: {e}", file=sys.stderr)
            print(f"Problematic content: {merged_review_content}", file=sys.stderr)
            all_results.append({
                "file_path": story_file_path_str,
                "error": f"Failed to parse merged review: {e}",
                "review": {}
            })

    # 全てのファイルの結果をJSONとして標準出力に出力
    print(json.dumps(all_results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
