import cv2, numpy as np, random, csv, threading

class videoTiles:
    # Coefficienti di peso
    #alpha, beta, gamma, delta = 0.4, 0.3, 0.2, 0.1
    alpha, beta, gamma, delta = 0.6, 0.3, 0.1, 0
    video_path = 'london360_1080p.mp4'
    movements_path = "Sampled_Logs_fast.txt"
    rows = 5  # Numero di righe di tiles
    cols = 9  # Numero di colonne di tiles
    movementsData=[[],[],[]] # Timestamp, MovimentiX, MovimentiY
    delayData=[]
    minP, maxP = 99, 0 # Minimo e massimo PSNR iniziali
    mosaic=None # Mosaico di tiles
    # Indicizzazione del file delle posizioni (definizione del ritardo)

    randomList=[[],[]] # Contiene gli spostamenti casuali
    infractionCounter=0
    # Valori di banda
    band=[]
    bandwidthMeanValue = 1000000.00 # byte
    #bandwidthMeanValue = 300000.00 # byte
    framesNumber = 897
    #framesNumber = 50
    # Variabili per PSNR e SIZE medi
    PSNRsum=0 # PSNR
    nPSNRsum=0 # PSNR normalizzato
    SIZEsum=0 # SIZE OTTENUTA pesando l'intero mosaic
    SIZECleanSum = 0 #SIZE OTTENUTA SOMMANDO LE singole size delle tile
    tileSize={} # Dizionario per side delle tiles

    # Variabili per stampa su file
    # PSNR
    # PSNR intera zona (tile per tile)
    psnr_zone1_list = []
    psnr_zone2_list = []
    psnr_zone3_list = []
    psnr_zone4_list = []
    # PSNR medio (per tile)
    psnr_zone1_avg_list = []
    psnr_zone2_avg_list = []
    psnr_zone3_avg_list = []
    psnr_zone4_avg_list = []
    # PSNR totale
    psnr_total_list = []
    # PSNR per calcolo media in ciascuna zona
    sumPSNRz1=0
    sumPSNRz2=0
    sumPSNRz3=0
    sumPSNRz4=0
    # SIZE
    # Size intera zona (tile per tile)
    size_zone1_list = []
    size_zone2_list = []
    size_zone3_list = []
    size_zone4_list = []
    # Size intera zona (cumulativa)
    size_zone1_sum_list = []
    size_zone2_sum_list = []
    size_zone3_sum_list = []
    size_zone4_sum_list = []
    # Size media (per tile)
    size_zone1_avg_list = []
    size_zone2_avg_list = []
    size_zone3_avg_list = []
    size_zone4_avg_list = []
    # Size per calcolo media in ciascuna zona
    sumSIZEz1=0
    sumSIZEz2=0
    sumSIZEz3=0
    sumSIZEz4=0
    # Size totale
    size_total_list = []
    
    # SOMMARIO
    summary_list = []

    # VARIABILI VISUALIZZAZIONE FRAME
    # Definisci una finestra specifica
    WINDOW_NAME = 'Frame'

    def __init__(self, delay):

        #Recupera dati di delay da delay.csv
        file_path = 'delay.csv'
        with open(file_path, 'r') as file:
            for line in file:
                data=line.strip() # Valore di delay
                floatData = (float(data))/10 # Valore di delay in float e proporzionato con divisione per 10
                backData = int(floatData/0.03) # Calcolo il valore del delay in frame (considerando 1 frame = 0.03 s, derivato dai 30fps del video)
                self.delayData.append(backData)
                #print(f"backData {backData}")
        self.delay = delay
        changeCounter = delay # Indice della posizione corrente 
        #GENERAZIONE BANDA

        #Genera valori di banda costanti
        self.band=self.fixedBandwidth(self.bandwidthMeanValue, self.framesNumber)

        #Genera valori di banda (60%)
        #self.band=self.lowBandwidth(self.bandwidthMeanValue, self.framesNumber)

        #Genera valori di banda (80%)
        #self.band=self.highBandwidth(self.bandwidthMeanValue, self.framesNumber)

         # Inizializza dizionario size delle tiles
        for i in range(self.rows*self.cols):
            self.tileSize[i] = 0

    def split_into_tiles(self,image, rows, cols):
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
    
    # Comprime le zone e aggiunge contorni colorati
    def modify_tile(self, tile, zone, cZ1, cZ2, cZ3, cZ4):
        modified_tile = tile.copy()
        if zone == 'z1':
            modified_tile=self.compress_image(tile, cZ1)
            modified_tile = self.add_border(tile,"red")
        elif zone == 'z2':
            modified_tile=self.compress_image(tile, cZ2)
            modified_tile = self.add_border(modified_tile,"green")
        elif zone == 'z3':
            modified_tile=self.compress_image(tile, cZ3)
            modified_tile = self.add_border(modified_tile,"blue")
        elif zone == 'z4':
            modified_tile = self.compress_image(tile, cZ4)
            modified_tile = self.add_border(modified_tile,"black")
        else:
            modified_tile = tile
        
        return modified_tile
    
    # Comprime le zone senza aggiungere contorni colorati
    def modify_tile_(self, tile, zone, cZ1, cZ2, cZ3, cZ4):
        modified_tile = tile.copy()
        if zone == 'z1':
            modified_tile=self.compress_image(tile, cZ1)
        elif zone == 'z2':
            modified_tile=self.compress_image(tile, cZ2)
        elif zone == 'z3':
            modified_tile=self.compress_image(tile, cZ3)
        elif zone == 'z4':
            modified_tile=self.compress_image(tile, cZ4)
        else:
            modified_tile = tile
        
        return modified_tile

    def add_border(self, frame, color="red", thickness=5):
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
    
    # Aggiunge un puntino al centro delle tile
    def add_center_dot(self, frame, color="white", radius=4):
        
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

    # Comanda l'aggiunta di un puntino al centro delle tile
    def modify_delay_tile(self, tile, zone):
        modified_tile = tile.copy()
        if zone == 'z1':
            modified_tile = self.add_center_dot(tile,"red",20)
            return modified_tile
        elif zone == 'z2':
            modified_tile = self.add_center_dot(tile,"green",16)
        elif zone == 'z3':
            modified_tile = self.add_center_dot(tile,"blue",12)
        elif zone == 'z4':
            modified_tile = self.add_center_dot(tile,"black")
        else:
            modified_tile = tile
        
        return modified_tile

    # Non modifica nulla nelle tile
    def modify_delay_tile_(self, tile, zone):
        return tile


    # Converte l'immagine in formato JPEG con una specifica qualità
    def compress_image(self, image, quality=100): #100 max (compressione 0), 1 min (compressione 100)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, compressed_image = cv2.imencode('.jpg', image, encode_param)
        decompressed_image = cv2.imdecode(compressed_image, 1)  # 1 per mantenere il formato a colori
        if quality == 100:
            return image
        return decompressed_image

    # PSNR STANDARD: Non tiene conto delle dimensioni delle immagini
    def calculate_psnr(self, original, compressed):
        mse = np.mean((original - compressed) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel_value = 255.0
        psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
        return psnr
    
    # Calcola la deviazione del viewport randomicamente
    def randomMoveViewPort(self):
        x_dev=random.randint(-3, 3)
        y_dev=random.randint(-2, 2)
        return x_dev,y_dev

    # Calcola la deviazione del viewport rispetto al centro in relazione ai dati x e y forniti
    def moveViewPort(self, xTiles, yTiles, x, y):
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

        #print(f"x: {x} y: {y} x_dev: {x_dev} y_dev: {y_dev}")
        return x_dev, y_dev
    
    # Mantiene fermo il viewport (nessun movimento dell'utente)
    def moveViewPortStill(self):
        x_dev = 0
        y_dev = 0
        return x_dev, y_dev

    # Aggiunge i movimenti alla lista
    def getFileCoords(self, file_csv):
        self.movementsData
        with open(file_csv, newline='') as csvfile:
            lettore_csv = csv.DictReader(csvfile)
            for riga in lettore_csv:
                self.movementsData[0].append(riga['timeStamp'])
                self.movementsData[1].append(riga['oriX'])
                self.movementsData[2].append(riga['oriZ'])

    # Aggiunge i movimenti (stazionari, cioè nessun movimento) alla lista
    def getFileCoords_(self, file_csv):
        self.movementsData
        for i in range(self.framesNumber):
                self.movementsData[0].append(0)
                self.movementsData[1].append(0)
                self.movementsData[2].append(0)        


    def goToFrame(self, cap, frame_number):
        # Verifica se il frame_number è valido
        if 0 <= frame_number < cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            return True
        else:
            print(f"Errore: Il frame {frame_number} non è valido.")
            return False

    def listToFile(self, lista, nome_file):
        with open(nome_file, 'w') as f:
            riga_formattata = ''  # Stringa temporanea per memorizzare i valori della riga corrente
            for elemento in lista:
                if elemento == '|':
                    # Se viene trovato '|', scrivi la riga formattata nel file e svuota la stringa temporanea
                    f.write(riga_formattata.rstrip() + '\n')
                    riga_formattata = ''  # Resetta la stringa temporanea per la prossima riga
                else:
                    # Aggiungi l'elemento alla riga formattata con una tabulazione
                    riga_formattata += str(elemento) + '\t'
            # Scrive l'ultima riga formattata nel file
            f.write(riga_formattata.rstrip() + '\n')

    def stringToFile(self, stringa, nome_file):
        # Apre il file in modalità append (se non esiste, lo crea)
        with open(nome_file, 'a') as f:
            # Scrive la stringa seguita da un nuovo capo
            f.write(stringa + '\n')

    def highBandwidth(self, valore_medio, numero_dati):
        rapportoValoreMedio = 0.8
        valore = valore_medio * rapportoValoreMedio
        dati_bandwidth = [random.uniform(valore_medio, valore) for _ in range(numero_dati)]
        return dati_bandwidth

    def lowBandwidth(self, valore_medio, numero_dati):
        rapportoValoreMedio = 0.6
        valore = valore_medio * rapportoValoreMedio
        dati_bandwidth = [random.uniform(valore_medio, valore) for _ in range(numero_dati)]
        return dati_bandwidth
    
    def fixedBandwidth(self, valore_medio, numero_dati):
        dati_bandwidth = []
        for _ in range(numero_dati):
            dati_bandwidth.append(valore_medio)
        return dati_bandwidth

    # Normalizza i valori in un intervallo [0,1]
    def normalize(self, valore):
        minimo = 29
        massimo = 50
        # Sogliatura
        if valore < minimo:
            valore = minimo
        if valore > massimo:
            valore = massimo
        return (valore - minimo) / (massimo - minimo)
    
    def frameWeight(self, frame):
        # Converte il frame in un array di byte
        _, encoded_frame = cv2.imencode('.jpg', frame)

        # Calcola la lunghezza dell'array di byte (peso del frame in byte)
        frame_weight = len(encoded_frame)

        return frame_weight

    def step(self, currentFrame, cZ1, cZ2, cZ3, cZ4):
                
                    
                    return self.mosaic,NORMALIZED_PSNR_FORMULA_DD, self.band[currentFrame+1], last



def display_frames(frameImage):
    while True:
        if frameImage is not None:
            cv2.imshow("Frame", frameImage)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()

def display_frame(frameImage):
    cv2.imshow("Frame", frameImage)

def display_frames_in_thread(frameImage):
    thread = threading.Thread(target=display_frames, args=(frameImage,))
    thread.start()
    return thread

def display_frame_in_thread(frameImage):
    thread = threading.Thread(target=display_frames, args=(frameImage,))
    thread.start()
    return thread

if __name__ == "__main__":
    WINDOW_NAME = 'Frame'
    v = videoTiles(9)  # Delay espresso in frame: 3: 100ms, 6: 200ms, 9: 300ms, 12: 400ms, 15: 500ms, ..., 30: 1000ms = 1s
    cap = cv2.VideoCapture(v.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    z1_quality = 100
    z2_quality = 70
    z3_quality = 30
    z4_quality = 0
    if not cap.isOpened():
        print("Errore: Impossibile aprire il video.")
 
    # Numero totale di frame nel video
    current_frame = 0

    # Visualizza il frame corrente
    frame=None

    while True:
        # Legge il frame corrente
        ret, frame = cap.read()

        if ret: #Se ret==True, il video è in corso
            
            #cv2.resizeWindow("Video", 400, 200)
            key = cv2.waitKey(1) & 0xFF
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
        
        else:
            print("Video terminato")
            break
        #encodedFrame = v.step(current_frame, z1_quality, z2_quality, z3_quality, z4_quality)
        #Importazione dati di movimento
        v.getFileCoords(v.movements_path)


        # Imposta il frame da visualizzare
        #if not v.goToFrame(cap, current_frame):
            #break

    
        # Peso il frame in byte
        weight = v.frameWeight(frame)
        #print(f"SIZE pre-compressione: {weight:.2f} byte -> {(weight/1024):.2f} kb -> {((weight/1024)/1024):.2f} mb")

        # Dividi il frame in tiles
        tiles = v.split_into_tiles(frame, v.rows, v.cols)

        #ZONA CENTRALE: Calcola le coordinate del centro dell'immagine
        central_point_x=v.cols//2
        central_point_y=v.rows//2
        #print(f"self.cols: {self.cols}, self.rows: {self.rows}\nself.cols//2: {self.cols//2}, self.rows//2: {self.rows//2}")

        timestamp=v.movementsData[0][current_frame]
        movementX=v.movementsData[1][current_frame]
        movementY=v.movementsData[2][current_frame]
        #print(f"changeCounter: {currentFrame} | timestamp: {timestamp}")

        
        # Sposta il centro del viewport in base ai movimenti dell'utente (da file)
        x_dev, y_dev=v.moveViewPort(v.rows,v.cols,movementX,movementY)
        # Sposta il centro del viewport in base ai movimenti dell'utente (random)
        #x_dev, y_dev=v.randomMoveViewPort()
        # Mantiene fermo il centro del viewport
        #x_dev, y_dev=v.moveViewPortStill()
        

        # Memorizzo la deviazione al frame corrente
        v.randomList[0].append(x_dev)
        v.randomList[1].append(y_dev)

        # Stampa deviazioni a ritardo 0
        #print(f"Deviazioni a ritardo 0")
        #print(f"x_dev: {x_dev}")
        #print(f"y_dev: {y_dev}\n")
        #print(f"self.delay: {self.delay}")

        # Si lavora solo quando il sistema a delay D può produrre un'immagine
        if current_frame >= v.delay:
            #self.delay = self.delayData[currentFrame] # Valori di delay presi da file, se commentato si ha delay costante pari a self.delay (impostato sopra)
            # Stampa le deviazioni a ritardo D
            #print(f"Deviazioni a ritardo D: {self.delay}")
            #print(f"current_frame-self.delay: {currentFrame-self.delay}")
            #print(f"x_dev: {self.randomList[0][currentFrame-self.delay]}")
            #print(f"y_dev: {self.randomList[1][currentFrame-self.delay]}\n")

            # RICALCOLO DEL CENTRO IN RELAZIONE ALLA DEVIAZIONE A RITARDO D
            central_point_x+=v.randomList[0][current_frame-v.delay]
            central_point_y+=v.randomList[1][current_frame-v.delay]

            # Inizializza i dizionari per tenere traccia dei PSNR per ogni zona
            psnr_zone1 = {'sum': 0, 'count': 0}
            psnr_zone2 = {'sum': 0, 'count': 0}
            psnr_zone3 = {'sum': 0, 'count': 0}
            psnr_zone4 = {'sum': 0, 'count': 0}

            # Modifica le singole tiles in base alla zona concentrica
            modTiles=tiles.copy()
            #print(f"central_point_x: {central_point_x}, central_point_y: {central_point_y}")
            for i, tile in enumerate(tiles):
                row, col = divmod(i, v.cols)
                #print(f"i: {i}, Row: {row}, Col: {col}")
                zone = 'z4'
                if central_point_y-2 <= row <= central_point_y+2 and central_point_x-2 <= col <= central_point_x+2 :
                    zone = 'z3'
                if central_point_y-1 <= row <= central_point_y+1 and central_point_x-1 <= col <= central_point_x+1 :
                    zone = 'z2'
                if row == central_point_y and col == central_point_x:
                    zone = 'z1'
                #print(f"Zone: {zone}")

                
                modified_tile = v.modify_tile(tile, zone, z1_quality, z2_quality, z3_quality, z4_quality)
                psnr = "{:.2f}".format(v.calculate_psnr(tile,modified_tile))
                #print(f"Tile: {i} | Zone: {zone} | PSNR: {psnr}")
                modTiles[i] = modified_tile

                # Calcolo PSNR per zona
                
                if zone == 'z1':
                    #print(f"psnr z1: {psnr}")
                    psnr_zone1['sum'] += float(psnr)
                    psnr_zone1['count'] += 1
                elif zone == 'z2':
                    #print(f"psnr z2: {psnr}")
                    psnr_zone2['sum'] += float(psnr)
                    psnr_zone2['count'] += 1
                elif zone == 'z3':
                    #print(f"psnr z3: {psnr}")
                    psnr_zone3['sum'] += float(psnr)
                    psnr_zone3['count'] += 1
                elif zone == 'z4':
                    #print(f"psnr z4: {psnr}")
                    psnr_zone4['sum'] += float(psnr)
                    psnr_zone4['count'] += 1
                
            # ----------------------------------------------------------
            # MOVIMENTI DELL'UTENTE IN TEMPO REALE

            # Calcolo (azzeramento) del centro
            central_point_x=v.cols//2
            central_point_y=v.rows//2
            #print(f"self.cols: {self.cols}, self.rows: {self.rows}\nself.cols//2: {self.cols//2}, self.rows//2: {self.rows//2}")
            ## RICALCOLO DEL CENTRO IN RELAZIONE ALLA DEVIAZIONE A RITARDO 0
            central_point_x+=x_dev
            central_point_y+=y_dev

            # Inizializza i dizionari per tenere traccia dei PSNR per ogni zona
            psnr_zone1 = {'sum': 0, 'count': 0}
            psnr_zone2 = {'sum': 0, 'count': 0}
            psnr_zone3 = {'sum': 0, 'count': 0}
            psnr_zone4 = {'sum': 0, 'count': 0}

            # Inizializza i dizionari per tenere traccia della size per ogni zona
            size_zone1 = {'sum': 0, 'count': 0}
            size_zone2 = {'sum': 0, 'count': 0}
            size_zone3 = {'sum': 0, 'count': 0}
            size_zone4 = {'sum': 0, 'count': 0}

            # Modifica le singole tiles in base alla zona concentrica
            #print(f"central_point_x: {central_point_x}, central_point_y: {central_point_y}")
            for i, tile in enumerate(tiles):
                row, col = divmod(i, v.cols)
                #print(f"i: {i}, Row: {row}, Col: {col}")
                zone = 'z4'
                if central_point_y-2 <= row <= central_point_y+2 and central_point_x-2 <= col <= central_point_x+2 :
                    zone = 'z3'
                if central_point_y-1 <= row <= central_point_y+1 and central_point_x-1 <= col <= central_point_x+1 :
                    zone = 'z2'
                if row == central_point_y and col == central_point_x:
                    zone = 'z1'
                #print(f"Zone: {zone}")
                
                modified_tile = v.modify_delay_tile(modTiles[i], zone)

                # Calcolo il PSNR della tile
                psnr = min(float("{:.2f}".format(v.calculate_psnr(tile, modified_tile))), 50)

                # Calcolo la size della tile
                size = "{:.2f}".format(v.frameWeight(modified_tile))

                tiles[i] = modified_tile


                # Calcolo PSNR per zona
                if zone == 'z1': # 1 tile
                    #print(f"psnr z1: {psnr}")
                    #print(f"size z1: {size}")
                    #v.psnr_zone1_list.append(psnr)
                    #v.size_zone1_list.append(size)
                    psnr_zone1['sum'] += float(psnr)
                    psnr_zone1['count'] += 1
                    size_zone1['sum'] += float(size)
                    size_zone1['count'] += 1
                    
                elif zone == 'z2': # 8 tiles
                    #print(f"psnr z2: {psnr}")
                    #print(f"size z2: {size}")
                    #v.psnr_zone2_list.append(psnr)
                    #v.size_zone2_list.append(size)
                    psnr_zone2['sum'] += float(psnr)
                    psnr_zone2['count'] += 1
                    size_zone2['sum'] += float(size)
                    size_zone2['count'] += 1
                    
                elif zone == 'z3': # 16 tiles
                    #print(f"psnr z3: {psnr}")
                    #print(f"size z3: {size}")
                    #v.psnr_zone3_list.append(psnr)
                    #v.size_zone3_list.append(size)
                    psnr_zone3['sum'] += float(psnr)
                    psnr_zone3['count'] += 1
                    size_zone3['sum'] += float(size)
                    size_zone3['count'] += 1
                    
                elif zone == 'z4': # 20 tiles
                    #print(f"psnr z4: {psnr}")
                    #print(f"size z4: {size}")
                    #v.psnr_zone4_list.append(psnr)
                    #v.size_zone4_list.append(size)
                    psnr_zone4['sum'] += float(psnr)
                    psnr_zone4['count'] += 1
                    size_zone4['sum'] += float(size)
                    size_zone4['count'] += 1
                    
                
                # Calcolo SIZE della TILE
                tileWeight = v.frameWeight(tiles[i]) #Calcolo il peso della tile

                cumulativeTileSize = v.tileSize[i] #Recupero la size cumulativa della tile
                updatedTileSize = cumulativeTileSize+tileWeight
                v.tileSize.update({i : updatedTileSize})
                #print(f"Peso cumulativo tile {i}: {updatedTileSize}")
            
            # Calcolo PSNR medio per zona
            d_avg_psnr_zone1 = psnr_zone1['sum'] / psnr_zone1['count'] if psnr_zone1['count'] > 0 else 0
            d_avg_psnr_zone2 = psnr_zone2['sum'] / psnr_zone2['count'] if psnr_zone2['count'] > 0 else 0
            d_avg_psnr_zone3 = psnr_zone3['sum'] / psnr_zone3['count'] if psnr_zone3['count'] > 0 else 0
            d_avg_psnr_zone4 = psnr_zone4['sum'] / psnr_zone4['count'] if psnr_zone4['count'] > 0 else 0

            PSNR_FORMULA_DD = v.alpha*d_avg_psnr_zone1 + v.beta*d_avg_psnr_zone2 + v.gamma*d_avg_psnr_zone3 + v.delta*d_avg_psnr_zone4
    
            #Aggiorno minimo e massimo
            if PSNR_FORMULA_DD < v.minP:
                v.minP = PSNR_FORMULA_DD
            if PSNR_FORMULA_DD > v.maxP:
                v.maxP = PSNR_FORMULA_DD

            # Ricostruisci l'immagine con i tiles modificati
            v.mosaic=np.vstack([np.hstack(tiles[i*v.cols:(i+1)*v.cols]) for i in range(v.rows)])
            cv2.imshow("Video",v.mosaic) #Video a schermo
                        
            # Calcola somme per medie finali
            
            v.SIZECleanSum+=size_zone1['sum']+size_zone2['sum']+size_zone3['sum']+size_zone4['sum']
            #print("----------------------------")
        
        
        #print(f"p: {p} | b: {b} | l: {l}")
        # Aggiorna il "segnalatore" dei frame già visti
        current_frame += 1
        print(current_frame)