import torch
import time
from torch.nn.utils.rnn import pad_sequence

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model, dataloader, loss_punt_inic, loss_punt_final, loss_caps, optimizer):
    model.train()
    total_loss = 0
    for batch in dataloader:
        X = batch['X'].to(DEVICE)
        lengths = batch['lengths'].to(DEVICE)
        y_inic = batch['p_inicial'].to(DEVICE)
        y_final = batch['p_final'].to(DEVICE)
        y_cap = batch['cap'].to(DEVICE)

        optimizer.zero_grad()

        out_inic, out_final, out_cap = model(X, lengths)

        loss1 = loss_punt_inic(out_inic.view(-1, out_inic.size(-1)), y_inic.view(-1))
        loss2 = loss_punt_final(out_final.view(-1, out_final.size(-1)), y_final.view(-1))
        loss3 = loss_caps(out_cap.view(-1, out_cap.size(-1)), y_cap.view(-1))

        loss = loss1 + loss2 + loss3
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

# def train_epoch(model, dataloader, loss_punt_inic, loss_punt_final, loss_caps, optimizer):
#     model.train()
#     total_loss = 0
#     for batch in dataloader:
#         X = batch['X'].to(DEVICE)
#         print("X shape:", X.shape)
#         y_inic = batch['p_inicial'].to(DEVICE)
#         y_final = batch['p_final'].to(DEVICE)
#         y_cap = batch['cap'].to(DEVICE)

#         optimizer.zero_grad()

#         out_inic, out_final, out_cap = model(X)
        
#         loss1 = loss_punt_inic(out_inic, y_inic)
#         loss2 = loss_punt_final(out_final, y_final)
#         loss3 = loss_caps(out_cap, y_cap)

#         loss = loss1 + loss2 + loss3
#         loss.backward()
        
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
#         optimizer.step()
        
#         total_loss += loss.item()
#     return total_loss / len(dataloader)


def evaluate_model(model, dataloader, loss_punt_inic, loss_punt_final, loss_caps):
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
                                
            loss1 = loss_punt_inic(out_inic, y_inic)
            loss2 = loss_punt_final(out_final, y_final)
            loss3 = loss_caps(out_cap, y_cap)

            loss = loss1 + loss2 + loss3
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def fit(model, train_dataloader, val_dataloader, loss_punt_inic, loss_punt_final, loss_caps, optimizer, NUM_EPOCHS=100):
    train_losses = []
    val_losses = []
    print("Iniciando entrenamiento...")
    print(f"Épocas: {NUM_EPOCHS}, Tamaño de lote: {train_dataloader.batch_size}")
    
    for epoch in range(NUM_EPOCHS):
        time_start = time.time()
        train_loss = train_epoch(model, train_dataloader, loss_punt_inic, loss_punt_final, loss_caps, optimizer)
        # print("Pérdida de entrenamiento:", train_loss)
        train_losses.append(train_loss)
        
        val_loss = evaluate_model(model, val_dataloader, loss_punt_inic, loss_punt_final, loss_caps)
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

def collate_fn(batch):
    """
    batch: lista de dicts con keys: 'X', 'p_inicial', 'p_final', 'cap'
    """
    # Extraer los elementos por separado
    Xs = [item['X'] for item in batch]
    y_inic = [item['p_inicial'] for item in batch]
    y_final = [item['p_final'] for item in batch]
    y_cap = [item['cap'] for item in batch]

    # Calcular longitudes originales
    lengths = torch.tensor([x.shape[0] for x in Xs])  # (batch,)

    # Padding (batch_size, max_seq_len, embedding_dim)
    Xs_padded = pad_sequence(Xs, batch_first=True, padding_value=0.0)

    # Padding de labels (batch_size, max_seq_len)
    y_inic_padded = pad_sequence(y_inic, batch_first=True, padding_value=-100)  # -100 se ignora en CrossEntropy
    y_final_padded = pad_sequence(y_final, batch_first=True, padding_value=-100)
    y_cap_padded = pad_sequence(y_cap, batch_first=True, padding_value=-100)

    return {
        'X': Xs_padded,                # (B, T, D)
        'p_inicial': y_inic_padded,   # (B, T)
        'p_final': y_final_padded,
        'cap': y_cap_padded,
        'lengths': lengths             # (B,)
    }