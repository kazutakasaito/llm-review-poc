"""
Multi-persona LLM reviewer - stable JP version (2025-05) - v4
--------------------------------------------------------
エンジニア / PdM / アーキテクトが JSON 形式で指摘を返し、
Supervisor がマージして review.json に出力します。
複数のストーリーファイルを処理できるように改善。
LLMが単一JSONオブジェクトを返した場合も配列として処理するように修正。
プロンプトを調整し、より多くの指摘を促すように変更。
"""

import json
import pathlib
import sys
import traceback # エラー詳細表示のため追加
from langchain_openai import ChatOpenAI
from langgraph.graph import MessageGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# ---------- Persona settings ----------
PERSONAS = {
    "engineer": dict(
        prompt=(
            "あなたは経験15年のシニアソフトウェアエンジニアです。\n"
            "ユーザーストーリーを実装観点（例：依存関係、技術的実現性、複雑度、テスト容易性、DoR達成状況、潜在的なバグのリスクなど）で多角的にレビューしてください。\n"
            "少なくとも3つの異なる具体的な指摘事項を挙げてください。より多くの指摘があれば、重要度順に含めてください。\n"
            "結果は必ずJSON配列 [{severity, comment, line}, ...] の形式で返してください。指摘が1件の場合でも必ず配列で囲ってください。\n"
            "severityは 'high', 'medium', 'low' のいずれかです。\n"
            "lineは指摘箇所の行番号(整数)ですが、特定できない場合はnullにしてください。"
        ),
        temperature=0.0, # 安定した出力を求めるため低めに設定
    ),
    "pdm": dict(
        prompt=(
            "あなたはプロダクトマネージャーです。\n"
            "ユーザーストーリーをプロダクト価値の観点（例：ビジネス目標への貢献、KPIへの影響、ユーザーストーリーの明確さ、受入基準の網羅性、市場適合性、競合との差別化など）で多角的にレビューしてください。\n"
            "少なくとも3つの異なる具体的な指摘事項を挙げてください。より多くの指摘があれば、重要度順に含めてください。\n"
            "結果は必ずJSON配列 [{severity, comment, line}, ...] の形式で返してください。指摘が1件の場合でも必ず配列で囲ってください。\n"
            "severityは 'high', 'medium', 'low' のいずれかです。\n"
            "lineは指摘箇所の行番号(整数)ですが、特定できない場合はnullにしてください。"
        ),
        temperature=0.3, # 多様な視点を得るため少し高めに設定
    ),
    "architect": dict(
        prompt=(
            "あなたはソフトウェアアーキテクトです。\n"
            "ユーザーストーリーを非機能要件の観点（例：パフォーマンス、スケーラビリティ、セキュリティ、可用性、保守性、技術的負債、既存システムとの整合性など）で多角的にレビューしてください。\n"
            "少なくとも3つの異なる具体的な指摘事項を挙げてください。より多くの指摘があれば、重要度順に含めてください。\n"
            "結果は必ずJSON配列 [{severity, comment, line}, ...] の形式で返してください。指摘が1件の場合でも必ず配列で囲ってください。\n"
            "severityは 'high', 'medium', 'low' のいずれかです。\n"
            "lineは指摘箇所の行番号(整数)ですが、特定できない場合はnullにしてください。"
        ),
        temperature=0.0, # 安定した出力を求めるため低めに設定
    ),
}

# ---------- LLM factory (JSON-mode ON) ----------
def make_llm(cfg):
    """LLMクライアントを生成します。JSONモードを有効にします。"""
    return ChatOpenAI(
        model="gpt-4o-mini",  # より多くの指摘を期待してモデルを変更 (元は gpt-3.5-turbo)
        temperature=cfg["temperature"],
        max_tokens=1536, # 複数の指摘を返すためにトークン上限を少し増やす
        model_kwargs={"response_format": {"type": "json_object"}},
    )

# ---------- Graph Definition ----------
def create_workflow():
    """レビューワークフローのグラフを構築します。"""
    graph_builder = MessageGraph()

    # 各ペルソナのノードを定義
    for persona_name, config in PERSONAS.items():
        llm = make_llm(config)
        
        # persona_node 関数のクロージャ問題を避けるため、デフォルト引数で現在の値をキャプチャ
        def create_persona_node_func(p_name, p_config, p_llm):
            def persona_node_func(messages):
                # messagesにはこれまでの会話履歴が入る。最後のメッセージが対象のストーリー。
                if not messages:
                    return AIMessage(role=p_name, content="[]")

                story_content = ""
                for msg in reversed(messages):
                    if isinstance(msg, HumanMessage) and msg.role == "user_story_content":
                        story_content = msg.content
                        break
                
                if not story_content:
                    print(f"Warning: Story content not found for persona {p_name}. Returning empty list.", file=sys.stderr)
                    return AIMessage(role=p_name, content="[]")

                full_prompt_messages = [
                    SystemMessage(content=p_config["prompt"]),
                    HumanMessage(content=f"--- 対象ストーリー ---\n{story_content}\n\n### 出力は指示されたJSONオブジェクトのみ ###")
                ]
                
                print(f"\n--- Sending to LLM for persona: {p_name} ---", file=sys.stderr)
                # print(f"Full prompt messages: {full_prompt_messages}", file=sys.stderr) # デバッグ用

                processed_content = "[]" # デフォルトは空の配列文字列
                try:
                    response = p_llm.invoke(full_prompt_messages)
                    
                    print(f"--- Raw LLM response for {p_name}: ---", file=sys.stderr)
                    print(response.content, file=sys.stderr)
                    print(f"--- End of raw LLM response for {p_name} ---", file=sys.stderr)

                    if response.content and response.content.strip():
                        content_strip = response.content.strip()
                        if content_strip.startswith("[") and content_strip.endswith("]"):
                            # 既に配列形式の場合
                            try:
                                json.loads(content_strip) # JSONとして正しいかパース試行
                                processed_content = content_strip
                            except json.JSONDecodeError as e_parse_array:
                                print(f"Warning: Failed to parse JSON array from {p_name}. Error: {e_parse_array}. Response was: '{response.content}'", file=sys.stderr)
                        elif content_strip.startswith("{") and content_strip.endswith("}"):
                            # 単一のJSONオブジェクトの場合、配列でラップする
                            print(f"Info: LLM returned a single JSON object for {p_name}. Wrapping in an array. Response was: '{response.content}'", file=sys.stderr)
                            try:
                                json.loads(content_strip) # まず単一オブジェクトとして正しいかパース試行
                                processed_content = f"[{content_strip}]" # 配列でラップ
                            except json.JSONDecodeError as e_parse_object:
                                print(f"Warning: Failed to parse single JSON object from {p_name}. Error: {e_parse_object}. Response was: '{response.content}'", file=sys.stderr)
                        else:
                            # それ以外の不正な形式の場合
                            print(f"Warning: Invalid JSON structure from {p_name}. Does not start with '[' or '{{'. Response was: '{response.content}'", file=sys.stderr)
                    else:
                        print(f"Warning: Empty response content from {p_name}.", file=sys.stderr)
                    
                    return AIMessage(role=p_name, content=processed_content)

                except Exception as e_invoke:
                    print(f"Error invoking LLM for {p_name}: {e_invoke}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                    return AIMessage(role=p_name, content="[]") # エラー時も空の配列

            return persona_node_func

        # 各ペルソナに対応するノード関数を作成してグラフに追加
        node_func = create_persona_node_func(persona_name, config, llm)
        graph_builder.add_node(persona_name, node_func)
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
                    persona_review_list = json.loads(msg.content) # processed_content は既に文字列のはず
                    merged_reviews[msg.role] = persona_review_list
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON in merge_node from {msg.role}: {msg.content}", file=sys.stderr)
                    merged_reviews[msg.role] = [{"severity": "error", "comment": f"Failed to parse review from {msg.role} in merge_node.", "line": None}]
                except Exception as e:
                    print(f"Error processing message in merge_node from {msg.role}: {e}", file=sys.stderr)
                    merged_reviews[msg.role] = [{"severity": "error", "comment": f"Unexpected error processing review from {msg.role} in merge_node.", "line": None}]

        return AIMessage(role="supervisor", content=json.dumps(merged_reviews, ensure_ascii=False))

    graph_builder.add_node("merge_reviews", merge_node)
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
        
        initial_message = [HumanMessage(content=story_text, role="user_story_content")]
        
        print(f"Reviewing {story_file_path_str}...", file=sys.stderr)
        
        final_state_messages = workflow.invoke(initial_message)
        
        merged_review_content = "{}" 
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

    print(json.dumps(all_results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
