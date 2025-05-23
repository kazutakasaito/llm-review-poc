# ユーザーストーリー: ユーザー登録をする

## ナラティブ
★ 新規ユーザーとして、メールアドレスとパスワードを入力してアカウントを作成したい。  
なぜなら、アプリのパーソナライズ機能を利用したいからだ。

## Definition of Ready
- [ ] このストーリーのスコープは 1 スプリント以内  
- [ ] 入力項目（メール、パスワード）のバリデーション要件が決まっている  
- [ ] バックエンドの `/signup` API が未実装であることをチームが共有している

## Definition of Done
- [ ] フロント：登録フォーム画面を実装し、必須入力チェックが通る  
- [ ] バックエンド：`/signup` POST エンドポイントが 201 を返す  
- [ ] 成功時にログイン状態となり、ユーザー名がヘッダーに表示される  
- [ ] Postman / e2e テストシナリオがリポジトリに追加される

## 受入条件（Gherkin）
```gherkin
Scenario: 正常にユーザーを登録する
  Given 未登録ユーザーがサインアップ画面を開いている
  When  メールに "test@example.com"、パスワードに "P@ssw0rd!" を入力して「登録」を押す
  Then  サーバーが 201 を返し、トップ画面にリダイレクトされ、
        ヘッダーに "test@example.com" が表示される
