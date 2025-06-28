import torch

def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        input_seq = batch['X']
        target_seq = batch['y']
        optimizer.zero_grad()
        # Forward pass
        output = model(input_seq)
        # Calcular pérdida
        # Reshape para calcular cross entropy
        output = output.reshape(-1, output.size(-1))

        loss = criterion(output, target_seq)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping para evitar exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()


def evaluate_model(model, dataloader, criterion):
    """Evalúa el modelo en el conjunto de validación"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_seq = batch['X']
            target_seq = batch['y']
            output = model(input_seq)  
            
            output = output.reshape(-1, output.size(-1))
            
            loss = criterion(output, target_seq)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def fit(model, train_dataloader, val_dataloader, criterion, optimizer, NUM_EPOCHS=100):
    train_losses = []
    val_losses = []
    print("Iniciando entrenamiento...")
    print(f"Épocas: {NUM_EPOCHS}, Tamaño de lote: {train_dataloader.batch_size}")
    for epoch in range(NUM_EPOCHS):
    # Entrenamiento
        train_loss = train_epoch(model, train_dataloader, criterion, optimizer)
        train_losses.append(train_loss)
        
        # Validación
        val_loss = evaluate_model(model, val_dataloader, criterion)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Época {epoch+1}/{NUM_EPOCHS}')
            print(f'  Pérdida Entrenamiento: {train_loss:.4f}')
            print(f'  Pérdida Validación: {val_loss:.4f}')
            print(f'  {"Mejorando" if val_loss < min(val_losses[:-1] + [float("inf")]) else "Empeorando"}')

    print("Entrenamiento completado!")
    return train_losses, val_losses