import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


@torch.no_grad
def get_batched_completions(
    model,
    tokenizer,
    embedding_list=None,
    token_list=None,
    max_new_tokens: int = 256,
    return_tokens=False,
) -> list[str] | torch.Tensor:
    """
    Generate completions for multiple prompts in a single batch.
    Heavily tested across models to be close to individual generations.
    This is far from trivial due to various padding (left/right) and masking issues.
    The final function is still not identical to individual generations, but it is close.
    The reason for this is that attention masks typically do not use -inf for masked tokens,
    but instead use values like -65504 for float16. This can lead to small differences in the
    final logits and thus the generated tokens.
    We are much closer to individual generations than HF model.generate, which often
    fails in mysterious ways.
    No KV-cache for now.

    Args:
        model: A pretrained model.
        tokenizer: A pretrained tokenizer.
        embedding_list: A list of embeddings for each prompt.
        token_list: A list of tokens for each prompt.
        max_new_tokens: The maximum number of tokens to generate for each prompt.
    Returns:
        A list of completions for each prompt.
    """
    assert embedding_list is not None or token_list is not None
    if embedding_list is not None:
        assert all(e.ndim == 2 for e in embedding_list)
    if token_list is not None:
        assert all(t.ndim == 1 for t in token_list)

    # We first pad the embeddings to the maximum context length of the model.
    # Positions after the current one will be ignored via the attention mask.
    if token_list is not None:
        embedding_list = [
            model.get_input_embeddings()(t.unsqueeze(0))[0] for t in token_list
        ]

    B = len(embedding_list)
    # Add left padding
    embeddings = pad_sequence(
        [e.flip(0) for e in embedding_list], batch_first=True, padding_value=0
    ).flip(1)
    padded_embeddings = F.pad(embeddings, (0, 0, 0, max_new_tokens))
    # Create attention mask and position ids
    lengths = [
        {
            "padding": embeddings.size(1) - e.size(0),
            "generation": padded_embeddings.size(1) - (embeddings.size(1) - e.size(0)),
        }
        for e in embedding_list
    ]
    attention_mask = torch.stack(
        [
            torch.cat([torch.zeros(pl["padding"]), torch.ones([pl["generation"]])])
            for pl in lengths
        ]
    ).to(model.device)
    position_ids = torch.stack(
        [
            torch.cat([torch.zeros(pl["padding"]), torch.arange(pl["generation"])])
            for pl in lengths
        ]
    ).long().to(model.device)
    tokens = []
    next_token_idx = embeddings.size(1)
    for i in range(max_new_tokens):
        outputs = model(
            inputs_embeds=padded_embeddings[:, :next_token_idx],
            attention_mask=attention_mask[:, :next_token_idx],
            position_ids=position_ids[:, :next_token_idx],
        )
        next_tokens = outputs.logits.argmax(dim=-1)[torch.arange(B), -1]
        tokens.append(next_tokens)
        padded_embeddings[torch.arange(B), next_token_idx] = (
            model.get_input_embeddings()(next_tokens).detach()
        )
        next_token_idx += 1

    tokens = torch.stack(tokens, dim=0).T
    if return_tokens:
        return tokens
    completion = tokenizer.batch_decode(tokens, skip_special_tokens=False)
    completion = [c.split(tokenizer.eos_token)[0] for c in completion]
    return completion
