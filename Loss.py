def loss_fn(start_logits, end_logits, start_positions, end_positions):
    """
    Return the sum of the cross entropy losses for both the start and end logits
    """
    loss_fct = nn.CrossEntropyLoss()
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = (start_loss + end_loss)
    return total_loss
