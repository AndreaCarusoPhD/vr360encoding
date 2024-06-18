import cv2
import numpy as np
import random
import csv

video_path = 'london360_1080p.mp4'
rows = 5  # Numero di righe di tiles
cols = 9  # Numero di colonne di tiles
movementsData=[[],[],[]] # Timestamp, MovimentiX, MovimentiY

# Indicizzazione del file delle posizioni (definizione del ritardo)
delay = 10 # Ritardo di delay righe
changeCounter = delay # Indice della posizione corrente
framesNumber = 897

def split_into_tiles(image, rows, cols):
    # Calcola le dimensioni dei tiles
    height, width, _ = image.shape
    tile_height = height // rows
    tile_width = width // cols

    tiles = []
    # Estrai i tiles dall'immagine
    for r in range(rows):
        for c in range(cols):
            tile = image[r * tile_height:(r + 1) * tile_height, c * tile_width:(c + 1) * tile_width]
            tiles.append(tile)

    return tiles

def modify_tile(tile, zone):
    modified_tile = tile.copy()
    # Esempio di modifica: inverti i colori per zone diverse
    if zone == 'z1':
        modified_tile = add_border(tile,"red")
        return modified_tile
    elif zone == 'z2':
        modified_tile=compress_image(tile,80)
        modified_tile = add_border(modified_tile,"green")
    elif zone == 'z3':
        modified_tile=compress_image(tile,10)
        modified_tile = add_border(modified_tile,"blue")
    elif zone == 'z4':
        modified_tile=compress_image(tile,0)
        modified_tile = add_border(modified_tile,"black")
    else:
        modified_tile = tile
    
    return modified_tile

def modify_tile_(tile, zone):
    modified_tile = tile.copy()
    # Esempio di modifica: inverti i colori per zone diverse
    if zone == 'z1':
        #modified_tile = add_border(tile,"red")
        return modified_tile
    elif zone == 'z2':
        modified_tile=compress_image(tile,80)
        #modified_tile = add_border(modified_tile,"green")
    elif zone == 'z3':
        modified_tile=compress_image(tile,10)
        #modified_tile = add_border(modified_tile,"blue")
    elif zone == 'z4':
        modified_tile=compress_image(tile,0)
        #modified_tile = add_border(modified_tile,"black")
    else:
        modified_tile = tile
    
    return modified_tile

def add_border(frame, color="red", thickness=5):
    # Aggiungi un contorno
    if color=="red":
        frame_with_border = cv2.rectangle(frame.copy(), (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), thickness)
    if color=="green":
        frame_with_border = cv2.rectangle(frame.copy(), (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), thickness)
    if color=="blue":
        frame_with_border = cv2.rectangle(frame.copy(), (0, 0), (frame.shape[1], frame.shape[0]), (255, 0, 0), thickness)
    if color=="black":
        frame_with_border = cv2.rectangle(frame.copy(), (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), thickness)
    return frame_with_border


def add_center_dot(frame, color="white", radius=4):
    # Aggiungi un puntino al centro della tile
    frame_with_dot = frame.copy()
    height, width, _ = frame.shape
    center = (width // 2, height // 2)

    if color == "red":
        cv2.circle(frame_with_dot, center, radius, (0, 0, 255), -1)  # -1 indica che il cerchio è pieno
    if color == "green":
        cv2.circle(frame_with_dot, center, radius, (0, 255, 0), -1)  # -1 indica che il cerchio è pieno
    if color == "blue":
        cv2.circle(frame_with_dot, center, radius, (255, 0, 0), -1)  # -1 indica che il cerchio è pieno
    if color == "black":
        cv2.circle(frame_with_dot, center, radius, (0, 0, 0), -1)  # -1 indica che il cerchio è pieno

    return frame_with_dot

def modify_delay_tile(tile, zone):
    modified_tile = tile.copy()
    if zone == 'z1':
        modified_tile = add_center_dot(tile,"red",20)
        return modified_tile
    elif zone == 'z2':
        modified_tile=compress_image(tile,80)
        modified_tile = add_center_dot(tile,"green",16)
    elif zone == 'z3':
        modified_tile=compress_image(tile,10)
        modified_tile = add_center_dot(tile,"blue",12)
    elif zone == 'z4':
        modified_tile=compress_image(tile,0)
        modified_tile = add_center_dot(tile,"black")
    else:
        modified_tile = tile
    
    return modified_tile

def modify_delay_tile_(tile, zone):
    return tile


def compress_image(image, quality=10): #100 max, 1 min
    # Converte l'immagine in formato JPEG con una specifica qualità
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, compressed_image = cv2.imencode('.jpg', image, encode_param)
    decompressed_image = cv2.imdecode(compressed_image, 1)  # 1 per mantenere il formato a colori
    return decompressed_image

# PSNR STANDARD: Non tiene conto delle dimensioni delle immagini
def calculate_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel_value = 255.0
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr

def randomMoveViewPort():
    x_dev=random.randint(-2, 2)
    y_dev=random.randint(-2, 2)
    return x_dev,y_dev

def moveViewPort(xTiles, yTiles, x, y):
    horizontalDegreesPerPosition = 360 / yTiles
    verticalDegreesPerPosition = 360 / xTiles
    x = float(x)
    y = float(y)

    # Calcola la deviazione rispetto al centro dell'immagine
    x_centered_dev = (x % 360) - (360 / 2)
    y_centered_dev = (y % 360) - (360 / 2)

    # Normalizza la deviazione ai gradi per posizione
    x_dev = int(x_centered_dev / horizontalDegreesPerPosition / 2)
    y_dev = int(y_centered_dev / verticalDegreesPerPosition / 2)

    print(f"x: {x} y: {y} x_dev: {x_dev} y_dev: {y_dev}")
    return x_dev, y_dev

# Mantiene fermo il viewport (nessun movimento dell'utente)
def moveViewPortStill():
    x_dev = 0
    y_dev = 0
    return x_dev, y_dev

def getFileCoords(file_csv):
    global movementsData
    with open(file_csv, newline='') as csvfile:
        lettore_csv = csv.DictReader(csvfile)
        for riga in lettore_csv:
            movementsData[0].append(riga['timeStamp'])
            movementsData[1].append(riga['oriY'])
            movementsData[2].append(riga['oriZ'])
        
        print(f'timeStamp: {movementsData[0]} oriY: {movementsData[1]}, oriZ: {movementsData[2]}')


    # Aggiunge i movimenti (stazionari, cioè nessun movimento) alla lista
def getFileCoords_(file_csv):
    movementsData
    for i in range(framesNumber):
            movementsData[0].append(0)
            movementsData[1].append(0)
            movementsData[2].append(0)        


def main(video_path, rows, cols):
    # Apri il video utilizzando OpenCV
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    getFileCoords("movefast.txt")

    if not cap.isOpened():
        print("Errore nell'apertura del video.")
        return

    #DEVIAZIONE DAL CENTRO
    #Velocità di modifica della posizione
    changeCounter=0
    x_dev=0 #Deviazione asse x
    y_dev=0 #Deviazione asse y

    while True:
        ret, frame = cap.read()

        if not ret:
            break
        # Dividi il frame in tiles
        tiles = split_into_tiles(frame, rows, cols)

        #ZONA CENTRALE: Calcola le coordinate del centro dell'immagine
        central_point_x=cols//2
        central_point_y=rows//2

        timestamp=movementsData[0][changeCounter]
        movementX=movementsData[1][changeCounter]
        movementY=movementsData[2][changeCounter]
        print(f"changeCounter: {changeCounter} | timestamp: {timestamp}")

        #Incremento il contatore
        changeCounter=changeCounter+1
        
        x_dev, y_dev=moveViewPort(rows,cols,movementX,movementY)
        #x_dev, y_dev=moveViewPortStill()
        #RICALCOLO DEL CENTRO IN RELAZIONE ALLA DEVIAZIONE
        central_point_x+=x_dev
        central_point_y+=y_dev

        # Inizializza i dizionari per tenere traccia dei PSNR per ogni zona
        psnr_zone1 = {'sum': 0, 'count': 0}
        psnr_zone2 = {'sum': 0, 'count': 0}
        psnr_zone3 = {'sum': 0, 'count': 0}
        psnr_zone4 = {'sum': 0, 'count': 0}

        # Modifica i singoli tiles in base alla zona concentrica
        modTiles=tiles.copy()
        for i, tile in enumerate(tiles):
            row, col = divmod(i, cols)
            zone = 'z4'
            if central_point_y-2 <= row <= central_point_y+2 and central_point_x-2 <= col <= central_point_x+2 :
                zone = 'z3'
            if central_point_y-1 <= row <= central_point_y+1 and central_point_x-1 <= col <= central_point_x+1 :
                zone = 'z2'
            if row == central_point_y and col == central_point_x:
                zone = 'z1'
            
            modified_tile = modify_tile(tile, zone)
            psnr = "{:.2f}".format(calculate_psnr(tile,modified_tile))
            #print(f"Tile: {i} | Zone: {zone} | PSNR: {psnr}")
            modTiles[i] = modified_tile

            # Calcolo PSNR per zona
            if zone == 'z1':
                psnr_zone1['sum'] += float(psnr)
                psnr_zone1['count'] += 1
            elif zone == 'z2':
                psnr_zone2['sum'] += float(psnr)
                psnr_zone2['count'] += 1
            elif zone == 'z3':
                psnr_zone3['sum'] += float(psnr)
                psnr_zone3['count'] += 1
            elif zone == 'z4':
                psnr_zone4['sum'] += float(psnr)
                psnr_zone4['count'] += 1
            

        avg_psnr_zone1 = psnr_zone1['sum'] / psnr_zone1['count'] if psnr_zone1['count'] > 0 else 0
        avg_psnr_zone2 = psnr_zone2['sum'] / psnr_zone2['count'] if psnr_zone2['count'] > 0 else 0
        avg_psnr_zone3 = psnr_zone3['sum'] / psnr_zone3['count'] if psnr_zone3['count'] > 0 else 0
        avg_psnr_zone4 = psnr_zone4['sum'] / psnr_zone4['count'] if psnr_zone4['count'] > 0 else 0

        # Stampa la media dei valori PSNR per ogni zona
        print(f"DELAY: 0")
        print(f"fps: {fps}")
        print(f"Average PSNR for Zone 1: {avg_psnr_zone1:.2f}")
        print(f"Average PSNR for Zone 2: {avg_psnr_zone2:.2f}")
        print(f"Average PSNR for Zone 3: {avg_psnr_zone3:.2f}")
        print(f"Average PSNR for Zone 4: {avg_psnr_zone4:.2f}\n")

        # ------------------------------------------------------------
        # Implementazione a ritardo D
        #ZONA CENTRALE: Calcola le coordinate del centro dell'immagine
        central_point_x=cols//2
        central_point_y=rows//2

        timestamp=movementsData[0][changeCounter-delay]
        movementX=movementsData[1][changeCounter-delay]
        movementY=movementsData[2][changeCounter-delay]
        print(f"timestamp: {timestamp}")
        #changeCounter=changeCounter+10
        
        x_dev, y_dev=moveViewPort(rows,cols,movementX,movementY)
        #x_dev, y_dev=moveViewPortStill()
        #RICALCOLO DEL CENTRO IN RELAZIONE ALLA DEVIAZIONE
        central_point_x+=x_dev
        central_point_y+=y_dev

        # Inizializza i dizionari per tenere traccia dei PSNR per ogni zona
        psnr_zone1 = {'sum': 0, 'count': 0}
        psnr_zone2 = {'sum': 0, 'count': 0}
        psnr_zone3 = {'sum': 0, 'count': 0}
        psnr_zone4 = {'sum': 0, 'count': 0}

        # Modifica i singoli tiles in base alla zona concentrica
        for i, tile in enumerate(tiles):
            row, col = divmod(i, cols)
            zone = 'z4'
            if central_point_y-2 <= row <= central_point_y+2 and central_point_x-2 <= col <= central_point_x+2 :
                zone = 'z3'
            if central_point_y-1 <= row <= central_point_y+1 and central_point_x-1 <= col <= central_point_x+1 :
                zone = 'z2'
            if row == central_point_y and col == central_point_x:
                zone = 'z1'
            
            modified_tile = modify_delay_tile(modTiles[i], zone)
            psnr = "{:.2f}".format(calculate_psnr(tile,modified_tile))
            #print(f"Tile: {i} | Zone: {zone} | PSNR: {psnr}")
            tiles[i] = modified_tile

            # Calcolo PSNR per zona
            if zone == 'z1':
                psnr_zone1['sum'] += float(psnr)
                psnr_zone1['count'] += 1
            elif zone == 'z2':
                psnr_zone2['sum'] += float(psnr)
                psnr_zone2['count'] += 1
            elif zone == 'z3':
                psnr_zone3['sum'] += float(psnr)
                psnr_zone3['count'] += 1
            elif zone == 'z4':
                psnr_zone4['sum'] += float(psnr)
                psnr_zone4['count'] += 1
            

        d_avg_psnr_zone1 = psnr_zone1['sum'] / psnr_zone1['count'] if psnr_zone1['count'] > 0 else 0
        d_avg_psnr_zone2 = psnr_zone2['sum'] / psnr_zone2['count'] if psnr_zone2['count'] > 0 else 0
        d_avg_psnr_zone3 = psnr_zone3['sum'] / psnr_zone3['count'] if psnr_zone3['count'] > 0 else 0
        d_avg_psnr_zone4 = psnr_zone4['sum'] / psnr_zone4['count'] if psnr_zone4['count'] > 0 else 0

        # Stampa la media dei valori PSNR per ogni zona
        print(f"DELAY: {delay}")
        print(f"fps: {fps}")
        print(f"Average PSNR for Zone 1: {d_avg_psnr_zone1:.2f}")
        print(f"Average PSNR for Zone 2: {d_avg_psnr_zone2:.2f}")
        print(f"Average PSNR for Zone 3: {d_avg_psnr_zone3:.2f}")
        print(f"Average PSNR for Zone 4: {d_avg_psnr_zone4:.2f}\n")

        # Stampa la media dei valori PSNR per ogni zona
        print(f"DIFFERENCE DELAY 0 vs DELAY {delay}")
        print(f"fps: {fps}")
        print(f"Average PSNR difference Zone 1: {(avg_psnr_zone1-d_avg_psnr_zone1):.2f}")
        print(f"Average PSNR difference Zone 2: {(avg_psnr_zone2-d_avg_psnr_zone2):.2f}")
        print(f"Average PSNR difference Zone 3: {(avg_psnr_zone3-d_avg_psnr_zone3):.2f}")
        print(f"Average PSNR difference Zone 4: {(avg_psnr_zone4-d_avg_psnr_zone4):.2f}\n")

        # Ricostruisci l'immagine con i tiles modificati
        mosaic=np.vstack([np.hstack(tiles[i*cols:(i+1)*cols]) for i in range(rows)])

        # Visualizza il frame modificato
        cv2.imshow('Modified Frame', mosaic)

        # Premi Esc per uscire
        if cv2.waitKey(30) & 0xFF == 27:
            break

    # Rilascia le risorse
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(video_path, rows, cols)