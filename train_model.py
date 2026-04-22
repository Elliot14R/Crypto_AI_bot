# .github/workflows/train_model.yml
# Trains model and uploads as GitHub Release asset (bypasses LFS completely)
name: Train ML Model

on:
  workflow_dispatch:
    inputs:
      note:
        description: 'Reason for retraining'
        required: false
        default: 'Manual retrain'

jobs:
  train:
    runs-on: ubuntu-latest
    timeout-minutes: 45
    permissions:
      contents: write   # needed to create releases and push

    steps:
      - name: Checkout repo
        uses: actions/checkout@v4
        with:
          lfs: false        # skip LFS — avoids the budget error entirely

      - name: Setup Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: pip install -q pandas numpy scikit-learn xgboost joblib requests

      - name: Train model
        run: |
          echo "Training started at $(date -u)"
          python train_model.py
          echo "Training done at $(date -u)"

      - name: Show results
        run: |
          python -c "
          import json
          p = json.load(open('model_performance.json'))
          print('='*50)
          print(f'ACCURACY:       {p[\"accuracy\"]}%')
          print(f'CV SCORE:       {p[\"cv_mean\"]}% ± {p[\"cv_std\"]}%')
          print(f'BUY precision:  {p.get(\"buy_precision\",\"?\")}%  recall: {p.get(\"buy_recall\",\"?\")}%')
          print(f'SELL precision: {p.get(\"sell_precision\",\"?\")}%  recall: {p.get(\"sell_recall\",\"?\")}%')
          print(f'TRAIN SET:      {p[\"n_train\"]} samples')
          print(f'TEST SET:       {p[\"n_test\"]} samples')
          print('='*50)
          "

      - name: Commit performance JSON (not the pkl)
        run: |
          git config user.name  "CryptoBot AI"
          git config user.email "bot@cryptobot.ai"
          git add model_performance.json
          git diff --staged --quiet || git commit -m "📊 Model retrained — $(python -c "import json; p=json.load(open('model_performance.json')); print(f'{p[\"accuracy\"]}% accuracy')")"
          git push
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

      - name: Upload model as GitHub Release asset
        # This stores the 51MB pkl outside of LFS/repo history
        # Render will download it at build time via GH_PAT_TOKEN
        run: |
          TAG="model-latest"
          REPO="${{ github.repository }}"
          TOKEN="${{ secrets.GITHUB_TOKEN }}"
          
          # Delete existing release if present
          RELEASE_ID=$(curl -s -H "Authorization: token $TOKEN" \
            "https://api.github.com/repos/$REPO/releases/tags/$TAG" | \
            python -c "import json,sys; d=json.load(sys.stdin); print(d.get('id',''))" 2>/dev/null)
          
          if [ -n "$RELEASE_ID" ] && [ "$RELEASE_ID" != "None" ]; then
            curl -s -X DELETE -H "Authorization: token $TOKEN" \
              "https://api.github.com/repos/$REPO/releases/$RELEASE_ID"
            echo "Deleted old release $RELEASE_ID"
          fi
          
          # Delete tag
          git tag -d $TAG 2>/dev/null || true
          git push origin :refs/tags/$TAG 2>/dev/null || true
          
          # Create new release
          RELEASE=$(curl -s -X POST \
            -H "Authorization: token $TOKEN" \
            -H "Content-Type: application/json" \
            "https://api.github.com/repos/$REPO/releases" \
            -d "{\"tag_name\":\"$TAG\",\"name\":\"Latest Model\",\"body\":\"Auto-updated model. Do not delete.\",\"draft\":false,\"prerelease\":false}")
          
          RELEASE_ID=$(echo $RELEASE | python -c "import json,sys; print(json.load(sys.stdin)['id'])")
          echo "Created release $RELEASE_ID"
          
          # Upload model file
          curl -s -X POST \
            -H "Authorization: token $TOKEN" \
            -H "Content-Type: application/octet-stream" \
            --data-binary @pro_crypto_ai_model.pkl \
            "https://uploads.github.com/repos/$REPO/releases/$RELEASE_ID/assets?name=pro_crypto_ai_model.pkl"
          
          echo "Model uploaded to GitHub Release: $TAG"

      - name: Upload as workflow artifact (backup)
        uses: actions/upload-artifact@v4
        with:
          name: model-${{ github.run_number }}
          path: |
            pro_crypto_ai_model.pkl
            model_performance.json
          retention-days: 30
