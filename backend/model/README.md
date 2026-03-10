# Place Your Model File Here

Copy your trained PyTorch model file (`.pt` or `.pth`) into this directory and rename it to `sarcasm_model.pt`

## Example:

```powershell
Copy-Item "path\to\your\trained_model.pt" -Destination "sarcasm_model.pt"
```

## Model Requirements

Your model should:
1. Be a PyTorch state dictionary (saved with `torch.save(model.state_dict(), 'model.pt')`)
2. Match the architecture defined in `../app.py`
3. Accept tokenized BERT inputs (input_ids and attention_mask)
4. Output logits for binary classification (2 classes: not sarcastic, sarcastic)

## Model Architecture Expected

- **Input**: BERT tokenized text (max_length=128)
- **Embedding**: BERT-base-uncased (768 dimensions)
- **CNN**: Conv1d layers with kernel_size=3, hidden_size=256
- **BiLSTM**: Bidirectional LSTM with hidden_size=256
- **Attention**: Multi-head attention with 8 heads
- **Output**: 2-class classification (binary)

If your model architecture is different, you'll need to modify the `SarcasmDetector` class in `app.py`.

## Testing Your Model

After placing your model file here, run:

```powershell
cd ..
python app.py
```

You should see: `Model loaded successfully on cpu` (or `cuda`)
