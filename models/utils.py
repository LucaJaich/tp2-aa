import torch
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    for batch in dataloader:
        X = batch['X'].to(DEVICE)
        y_inic = batch['p_inicial'].to(DEVICE)
        y_final = batch['p_final'].to(DEVICE)
        y_cap = batch['cap'].to(DEVICE)

        optimizer.zero_grad()

        out_inic, out_final, out_cap = model(X)
        
        loss1 = criterion(out_inic, y_inic)
        loss2 = criterion(out_final, y_final)
        loss3 = criterion(out_cap, y_cap)

        loss = loss1 + loss2 + loss3
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate_model(model, dataloader, criterion):
    """Evalúa el modelo en el conjunto de validación"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            X = batch['X'].to(DEVICE)
            y_inic = batch['p_inicial'].to(DEVICE)
            y_final = batch['p_final'].to(DEVICE)
            y_cap = batch['cap'].to(DEVICE)
            out_inic, out_final, out_cap = model(X)
                                
            loss1 = criterion(out_inic, y_inic)
            loss2 = criterion(out_final, y_final)
            loss3 = criterion(out_cap, y_cap)

            loss = loss1 + loss2 + loss3
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def fit(model, train_dataloader, val_dataloader, criterion, optimizer, NUM_EPOCHS=100):
    train_losses = []
    val_losses = []
    print("Iniciando entrenamiento...")
    print(f"Épocas: {NUM_EPOCHS}, Tamaño de lote: {train_dataloader.batch_size}")
    
    for epoch in range(NUM_EPOCHS):
        time_start = time.time()
        train_loss = train_epoch(model, train_dataloader, criterion, optimizer)
        # print("Pérdida de entrenamiento:", train_loss)
        train_losses.append(train_loss)
        
        val_loss = evaluate_model(model, val_dataloader, criterion)
        # print("Pérdida de VAL:", val_loss)

        val_losses.append(val_loss)
        
        print(f'Época {epoch+1}/{NUM_EPOCHS} - Pérdida Entrenamiento: {train_loss:.4f}, Pérdida Validación: {val_loss:.4f}')
        elapsed_time = time.time() - time_start
        print(f'Tiempo de la época: {elapsed_time:.2f} segundos')
        if (epoch + 1) % 10 == 0:
            print(f'Época {epoch+1}/{NUM_EPOCHS}')
            print(f'  Pérdida Entrenamiento: {train_loss:.4f}')
            print(f'  Pérdida Validación: {val_loss:.4f}')
            print(f'  {"Mejorando" if val_loss < min(val_losses[:-1] + [float("inf")]) else "Empeorando"}')

    print("Entrenamiento completado!")
    return train_losses, val_losses