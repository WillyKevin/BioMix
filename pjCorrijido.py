# -*- coding: utf-8 -*-
import streamlit as st
import cv2
import numpy as np
import os
from scipy import ndimage
from skimage.feature import peak_local_max
import pandas as pd
from PIL import Image
import time
import io # Required for in-memory files
import zipfile # Required for zipping
import datetime # Required for timestamping

# --- Configuração da Página Streamlit ---
st.set_page_config(layout="wide", page_title="Contador de Colônias")

# --- Funções Auxiliares ---

def recortar_placa(img_cv):
    """
    Tenta detectar a placa de Petri circular na imagem, criar uma máscara
    e recortar a imagem original para conter apenas a placa com uma pequena margem.
    Retorna a imagem recortada e True em caso de sucesso,
    ou a imagem original e False em caso de falha na detecção.
    """
    if len(img_cv.shape) == 3 and img_cv.shape[2] == 3:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    elif len(img_cv.shape) == 2:
        gray = img_cv
    elif len(img_cv.shape) == 3 and img_cv.shape[2] == 4: # Caso tenha canal alfa
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2GRAY)
    else:
        st.warning("Formato de imagem não suportado para detecção de placa.")
        return img_cv, False

    try:
        # Aumentar um pouco o blur pode ajudar em imagens ruidosas
        blur = cv2.medianBlur(gray, 21) # Kernel ímpar, aumentado

        # Ajustar parâmetros HoughCircles pode ser necessário dependendo das imagens
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, dp=1.3, minDist=gray.shape[0]//3, # Aumentado minDist
            param1=70, param2=50, # Ajustados params
            minRadius=max(80, gray.shape[0]//8), # Raio mínimo menor
            maxRadius=gray.shape[0]//2 + int(gray.shape[0]*0.1) # Raio máximo um pouco maior
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # Ordena por raio e talvez por proximidade ao centro? (opcional)
            # Apenas maior raio por enquanto
            circles = sorted(circles, key=lambda x: x[2], reverse=True)
            (x, y, r) = circles[0]

            # Verifica se o círculo detectado faz sentido (ex: não muito perto da borda)
            h, w = gray.shape
            if x < r * 0.5 or x > w - r * 0.5 or y < r * 0.5 or y > h - r * 0.5:
                 # print("Círculo detectado muito perto da borda, ignorando.")
                 return img_cv, False # Considera falha se o círculo for ruim

            mask = np.zeros_like(gray)
            cv2.circle(mask, (x, y), r, 255, -1)

            # Aplica a máscara na imagem original (que pode ser colorida ou cinza)
            if len(img_cv.shape) == 3:
                img_masked = cv2.bitwise_and(img_cv, img_cv, mask=mask)
            elif len(img_cv.shape) == 2:
                img_masked = cv2.bitwise_and(img_cv, img_cv, mask=mask)
            else: # Não deveria acontecer aqui devido à checagem inicial
                 return img_cv, False

            # Recorta usando as coordenadas do círculo
            margin = int(r * 0.05) # 5% de margem
            y_start = max(0, y - r - margin)
            y_end = min(img_masked.shape[0], y + r + margin)
            x_start = max(0, x - r - margin)
            x_end = min(img_masked.shape[1], x + r + margin)

            img_cortada = img_masked[y_start:y_end, x_start:x_end]

            # Verifica se o corte resultou em uma imagem válida
            if img_cortada.size == 0 or img_cortada.shape[0] < 10 or img_cortada.shape[1] < 10:
                # print("Corte resultou em imagem muito pequena ou vazia.")
                return img_cv, False

            return img_cortada, True
        else:
            # print("Nenhum círculo detectado pelo HoughCircles.")
            return img_cv, False
    except Exception as e:
        st.error(f"Erro inesperado em recortar_placa: {e}")
        return img_cv, False


def opencv_img_to_bytes(img_cv, format='PNG'):
    """Converte uma imagem OpenCV (numpy array) para bytes no formato especificado."""
    img_pil = None
    try:
        if len(img_cv.shape) == 3:
            if img_cv.shape[2] == 3: # BGR
                img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            elif img_cv.shape[2] == 4: # BGRA
                img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGBA)) # Use RGBA for PIL
            else: # Should not happen often
                 img_pil = Image.fromarray(img_cv) # Try direct conversion
        elif len(img_cv.shape) == 2: # Grayscale
            img_pil = Image.fromarray(img_cv, mode='L')
        else:
            raise ValueError(f"Formato de imagem OpenCV não suportado para conversão: shape={img_cv.shape}")

        if img_pil is None:
             raise ValueError("Falha ao converter imagem OpenCV para PIL.")

        byte_arr = io.BytesIO()
        img_pil.save(byte_arr, format=format)
        return byte_arr.getvalue()

    except Exception as e:
        st.error(f"Erro ao converter imagem OpenCV para bytes (Formato: {format}, Shape: {img_cv.shape}): {e}")
        return None


# --- Funções de Análise Específicas ---

def analisar_com_watershed(img_processar_bgr, area_minima, peak_footprint_size):
    """ Lógica de análise usando Watershed com marcadores de picos locais. """
    debug_images = {}
    img_processar_rgb = None # Inicializa

    # Garante que a imagem processada está em RGB para debug display
    try:
        if len(img_processar_bgr.shape) == 3 and img_processar_bgr.shape[2] == 3:
            img_processar_rgb = cv2.cvtColor(img_processar_bgr, cv2.COLOR_BGR2RGB)
        elif len(img_processar_bgr.shape) == 2:
            img_processar_rgb = cv2.cvtColor(img_processar_bgr, cv2.COLOR_GRAY2RGB)
        elif len(img_processar_bgr.shape) == 3 and img_processar_bgr.shape[2] == 4:
             img_processar_rgb = cv2.cvtColor(img_processar_bgr, cv2.COLOR_BGRA2RGB)
        else:
            img_processar_rgb = img_processar_bgr # Mantem se formato desconhecido
    except Exception as e:
        # print(f"Erro ao converter imagem de entrada do watershed para RGB: {e}")
        img_processar_rgb = img_processar_bgr # Fallback

    debug_images['imagem_entrada_watershed'] = img_processar_rgb

    # Garante BGR para processamento interno do Watershed
    if len(img_processar_bgr.shape) == 2:
         try:
            img_processar_bgr = cv2.cvtColor(img_processar_bgr, cv2.COLOR_GRAY2BGR)
         except Exception as e:
             st.error("Erro interno: Falha ao converter imagem cinza para BGR no Watershed.")
             return img_processar_rgb, 0, debug_images # Retorna RGB para exibição
    elif len(img_processar_bgr.shape) == 3 and img_processar_bgr.shape[2] == 4:
        try:
             img_processar_bgr = cv2.cvtColor(img_processar_bgr, cv2.COLOR_BGRA2BGR)
        except Exception as e:
             st.error("Erro interno: Falha ao converter imagem BGRA para BGR no Watershed.")
             return img_processar_rgb, 0, debug_images
    elif not (len(img_processar_bgr.shape) == 3 and img_processar_bgr.shape[2] == 3):
         st.error(f"Erro interno: Watershed esperava imagem BGR, recebeu shape {img_processar_bgr.shape}.")
         return img_processar_rgb, 0, debug_images

    # --- Lógica Watershed ---
    try:
        gray = cv2.cvtColor(img_processar_bgr, cv2.COLOR_BGR2GRAY)
        # GaussianBlur pode ser mais robusto que median para certas imagens
        blur = cv2.GaussianBlur(gray, (11, 11), 0)
        debug_images['blur_watershed'] = cv2.cvtColor(blur, cv2.COLOR_GRAY2RGB)

        # Thresholding adaptativo pode ser melhor em alguns casos
        # _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 15, 4) # Parâmetros podem precisar de ajuste
        debug_images['binarizada'] = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

        # Morfologia: Abrir para remover ruído, depois dilatar pode ajudar a conectar
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        # sure_bg = cv2.dilate(opening, kernel, iterations=3) # Área de fundo mais certa (opcional)
        debug_images['morfologia_opening'] = cv2.cvtColor(opening, cv2.COLOR_GRAY2RGB)

        # Transformada de distância na imagem aberta (objetos mais separados)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        dist_display = cv2.normalize(dist_transform, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        dist_display_color = cv2.applyColorMap(dist_display, cv2.COLORMAP_JET) # Usar JET ou outro
        debug_images['dist_transform'] = dist_display_color

        # Encontrar picos locais na transformada de distância
        if peak_footprint_size % 2 == 0: peak_footprint_size += 1 # Garante ímpar
        footprint_kernel = np.ones((peak_footprint_size, peak_footprint_size), dtype=bool)
        # min_distance ajuda a evitar múltiplos marcadores na mesma colônia
        coords = peak_local_max(dist_transform.astype(float), footprint=footprint_kernel, exclude_border=False, min_distance=max(5, peak_footprint_size // 2))

        if coords.shape[0] == 0:
            st.warning("Watershed: Nenhum pico local detectado para marcadores.")
            return img_processar_bgr, 0, debug_images # Retorna BGR original

        # Criar máscara e marcadores a partir dos picos
        local_max_mask = np.zeros(dist_transform.shape, dtype="uint8")
        local_max_mask[tuple(coords.T)] = 255
        debug_images['picos_locais_marcadores'] = cv2.cvtColor(local_max_mask, cv2.COLOR_GRAY2RGB)

        markers, num_markers = ndimage.label(local_max_mask)
        markers = markers + 1 # Adiciona 1 para que o fundo (0) não seja um marcador
        # Identificar região desconhecida (entre fundo e marcadores certos)
        # Usar a imagem 'opening' como base para 'unknown'
        unknown = cv2.subtract(opening, local_max_mask)
        markers[unknown == 255] = 0 # Marcar região desconhecida como 0 para watershed
        debug_images['regiao_desconhecida'] = cv2.cvtColor(unknown, cv2.COLOR_GRAY2RGB)

        # Aplicar Watershed
        labels = cv2.watershed(img_processar_bgr, markers.copy())

        # Processar resultado do Watershed
        resultado_final = img_processar_bgr.copy()
        contagem = 0
        ids_unicos = np.unique(labels)

        # Desenhar contornos e contar
        for label in ids_unicos:
            # Ignora fundo (-1) e região desconhecida (0) e bordas watershed (1)
            if label <= 1: continue

            mask = np.zeros(gray.shape, dtype="uint8")
            mask[labels == label] = 255

            cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not cnts: continue

            cnt = max(cnts, key=cv2.contourArea) # Pega o maior contorno para a label
            area = cv2.contourArea(cnt)

            if area > area_minima:
                contagem += 1
                # Desenha contorno verde
                cv2.drawContours(resultado_final, [cnt], -1, (0, 255, 0), 2)
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # Adiciona número azul perto do centro
                    cv2.putText(resultado_final, str(contagem), (cx - 10, cy + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2) # Ajuste fonte/cor

        # Adiciona contorno vermelho para as bordas do watershed (opcional)
        resultado_final[labels == -1] = [0, 0, 255] # Onde watershed desenhou bordas (-1) -> Vermelho
        debug_images['watershed_labels_colored'] = resultado_final

        return resultado_final, contagem, debug_images

    except Exception as e:
        st.error(f"Erro durante processamento Watershed: {e}")
        # Retorna a imagem BGR original que entrou na função e 0 contagem
        return img_processar_bgr, 0, debug_images


def analisar_com_blob_detector(img_processar_bgr, area_minima, min_circularity, min_convexity, blob_color):
    """ Lógica de análise usando SimpleBlobDetector. """
    debug_images = {}
    img_processar_rgb = None # Inicializa

    # Garante RGB para debug display
    try:
        if len(img_processar_bgr.shape) == 3 and img_processar_bgr.shape[2] == 3:
            img_processar_rgb = cv2.cvtColor(img_processar_bgr, cv2.COLOR_BGR2RGB)
        elif len(img_processar_bgr.shape) == 2:
            img_processar_rgb = cv2.cvtColor(img_processar_bgr, cv2.COLOR_GRAY2RGB)
        elif len(img_processar_bgr.shape) == 3 and img_processar_bgr.shape[2] == 4:
             img_processar_rgb = cv2.cvtColor(img_processar_bgr, cv2.COLOR_BGRA2RGB)
        else:
            img_processar_rgb = img_processar_bgr
    except Exception as e:
        # print(f"Erro ao converter imagem de entrada do blob para RGB: {e}")
        img_processar_rgb = img_processar_bgr # Fallback

    debug_images['imagem_entrada_blob'] = img_processar_rgb

    # Garante BGR para processamento e desenho
    if len(img_processar_bgr.shape) == 2:
         try:
             img_processar_bgr = cv2.cvtColor(img_processar_bgr, cv2.COLOR_GRAY2BGR)
         except Exception as e:
             st.error("Erro interno: Falha ao converter imagem cinza para BGR no Blob Detector.")
             return img_processar_rgb, 0, debug_images
    elif len(img_processar_bgr.shape) == 3 and img_processar_bgr.shape[2] == 4:
         try:
             img_processar_bgr = cv2.cvtColor(img_processar_bgr, cv2.COLOR_BGRA2BGR)
         except Exception as e:
             st.error("Erro interno: Falha ao converter imagem BGRA para BGR no Blob Detector.")
             return img_processar_rgb, 0, debug_images
    elif not (len(img_processar_bgr.shape) == 3 and img_processar_bgr.shape[2] == 3):
         st.error(f"Erro interno: Blob Detector esperava imagem BGR, recebeu shape {img_processar_bgr.shape}.")
         return img_processar_rgb, 0, debug_images


    # --- Lógica Blob Detector ---
    try:
        gray = cv2.cvtColor(img_processar_bgr, cv2.COLOR_BGR2GRAY)
        # Usar blur menor para Blob pode preservar melhor as bordas
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        debug_images['blur_blob'] = cv2.cvtColor(blur, cv2.COLOR_GRAY2RGB)

        # Thresholding: Otsu é bom, mas testar adaptativo pode valer a pena
        thresh_type = cv2.THRESH_BINARY_INV if blob_color == 0 else cv2.THRESH_BINARY
        _, binary = cv2.threshold(blur, 0, 255, thresh_type + cv2.THRESH_OTSU)
        # Alternativa Adaptativa:
        # binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                               thresh_type, 11, 2)
        debug_images['binarizada_blob'] = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

        # Configuração do Blob Detector
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = area_minima
        # params.maxArea = area_maxima # Pode adicionar um limite máximo se necessário

        params.filterByCircularity = True
        params.minCircularity = min_circularity
        # params.maxCircularity = 1.0 # Geralmente não limita o máximo

        params.filterByConvexity = True
        params.minConvexity = min_convexity
        # params.maxConvexity = 1.0

        params.filterByInertia = False # Geralmente não é tão útil para colônias
        # params.minInertiaRatio = 0.1

        params.filterByColor = True
        params.blobColor = blob_color # 0 para escuro, 255 para claro

        # Criar o detector e detectar blobs
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(binary) # Detecta na imagem binária

        # Desenhar resultados
        resultado_final = img_processar_bgr.copy() # Desenha na imagem BGR
        contagem = len(keypoints)

        # Imagem para debug mostrando keypoints na imagem original (RGB)
        img_with_keypoints_rgb = cv2.drawKeypoints(img_processar_rgb.copy(), keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        debug_images['keypoints_detectados'] = img_with_keypoints_rgb

        # Desenhar círculos e números na imagem de resultado final (BGR)
        for i, kp in enumerate(keypoints):
            x, y = int(kp.pt[0]), int(kp.pt[1])
            r = int(kp.size / 2)
            # Desenha círculo verde
            cv2.circle(resultado_final, (x, y), r, (0, 255, 0), 2)
            # Adiciona número azul
            cv2.putText(resultado_final, str(i + 1), (x - r // 2, y + r // 3), # Posição relativa ao tamanho
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2) # Ajuste fonte/cor

        return resultado_final, contagem, debug_images

    except Exception as e:
        st.error(f"Erro durante processamento Blob Detector: {e}")
        return img_processar_bgr, 0, debug_images


# --- Função Principal de Análise (Dispatcher) ---

def analisar_colonias(img_cv, area_minima, peak_footprint_size, analysis_method, blob_min_circularity, blob_min_convexity, blob_color):
    start_time = time.time()
    all_debug_data = {}
    analysis_date = datetime.datetime.now() # Captura data/hora da análise

    # 0. Imagem Original para Debug (Convertida para RGB)
    try:
        if len(img_cv.shape) == 3 and img_cv.shape[2] == 4:
            all_debug_data['imagem_original'] = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGB)
        elif len(img_cv.shape) == 3 and img_cv.shape[2] == 3:
             all_debug_data['imagem_original'] = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        elif len(img_cv.shape) == 2:
             all_debug_data['imagem_original'] = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
        else:
             all_debug_data['imagem_original'] = img_cv # Fallback
    except Exception as e:
         st.warning(f"Não foi possível converter imagem original para debug RGB: {e}")
         all_debug_data['imagem_original'] = img_cv

    # 1. Tenta recortar
    img_recortada_ou_original, placa_detectada = recortar_placa(img_cv.copy())
    all_debug_data['placa_detectada'] = placa_detectada

    # Converte resultado do recorte para RGB para debug ANTES de passar adiante
    try:
        if len(img_recortada_ou_original.shape) == 3:
             img_recorte_rgb = cv2.cvtColor(img_recortada_ou_original, cv2.COLOR_BGR2RGB if img_recortada_ou_original.shape[2]==3 else cv2.COLOR_BGRA2RGB)
        elif len(img_recortada_ou_original.shape) == 2:
             img_recorte_rgb = cv2.cvtColor(img_recortada_ou_original, cv2.COLOR_GRAY2RGB)
        else:
             img_recorte_rgb = img_recortada_ou_original # fallback
        all_debug_data['recorte_tentativa'] = img_recorte_rgb
    except Exception as e_cvt:
        st.warning(f"Não foi possível converter resultado do recorte para debug RGB: {e_cvt}")
        all_debug_data['recorte_tentativa'] = img_recortada_ou_original # fallback

    img_para_analise = img_recortada_ou_original

    # 2. Garante formato BGR para funções de análise (que esperam BGR)
    img_para_analise_bgr = None
    try:
        if len(img_para_analise.shape) == 3 and img_para_analise.shape[2] == 4: # BGRA
            img_para_analise_bgr = cv2.cvtColor(img_para_analise, cv2.COLOR_BGRA2BGR)
        elif len(img_para_analise.shape) == 2: # Cinza
            img_para_analise_bgr = cv2.cvtColor(img_para_analise, cv2.COLOR_GRAY2BGR)
        elif len(img_para_analise.shape) == 3 and img_para_analise.shape[2] == 3: # Já BGR
            img_para_analise_bgr = img_para_analise
        else:
            st.error(f"Formato de imagem inválido após tentativa de corte: {img_para_analise.shape}")
            # Retorna imagem original BGR se possível, ou a recortada como fallback
            try:
                 img_fallback_return = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2BGR) if img_cv.shape[2]==4 else (cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR) if len(img_cv.shape)==2 else img_cv)
                 if not (len(img_fallback_return.shape) == 3 and img_fallback_return.shape[2] == 3): # Final check
                     img_fallback_return = np.zeros((100,100,3), dtype=np.uint8) # Empty BGR image
            except:
                 img_fallback_return = np.zeros((100,100,3), dtype=np.uint8) # Empty BGR image

            return img_fallback_return, 0, 0.0, all_debug_data, "Erro de formato", "N/A", analysis_date.strftime("%Y-%m-%d %H:%M:%S")

        # Guarda a imagem que efetivamente entrou na análise para debug (convertida para RGB)
        all_debug_data['imagem_para_analise'] = cv2.cvtColor(img_para_analise_bgr, cv2.COLOR_BGR2RGB)

    except Exception as e_fmt:
        st.error(f"Erro ao preparar formato BGR da imagem para análise: {e_fmt}")
        try:
            img_fallback_return = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2BGR) if img_cv.shape[2]==4 else (cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR) if len(img_cv.shape)==2 else img_cv)
            if not (len(img_fallback_return.shape) == 3 and img_fallback_return.shape[2] == 3):
                 img_fallback_return = np.zeros((100,100,3), dtype=np.uint8)
        except:
             img_fallback_return = np.zeros((100,100,3), dtype=np.uint8)
        return img_fallback_return, 0, 0.0, all_debug_data, "Erro de formato", "N/A", analysis_date.strftime("%Y-%m-%d %H:%M:%S")


    # 3. Chama a função de análise específica
    try:
        if analysis_method == "Watershed (Picos Locais)":
            # Validação e ajuste dos parâmetros
            if peak_footprint_size is None or peak_footprint_size < 3: peak_footprint_size = 3
            elif peak_footprint_size % 2 == 0: peak_footprint_size += 1 # Garante ímpar >= 3

            resultado_final, contagem, debug_data_ws = analisar_com_watershed(
                img_para_analise_bgr, area_minima, peak_footprint_size
            )
            all_debug_data.update(debug_data_ws)
            method_used = "Watershed"
            parameters_used_info = f"Área mín: {area_minima}px | Vizinhança Marcadores: {peak_footprint_size}px"

        elif analysis_method == "Detector de Blobs":
            # Validação e ajuste dos parâmetros
            if blob_min_circularity is None or not (0.0 <= blob_min_circularity <= 1.0): blob_min_circularity = 0.0
            if blob_min_convexity is None or not (0.0 <= blob_min_convexity <= 1.0): blob_min_convexity = 0.0
            if blob_color is None or blob_color not in [0, 255]: blob_color = 0 # Default para preto

            resultado_final, contagem, debug_data_blob = analisar_com_blob_detector(
                img_para_analise_bgr, area_minima, blob_min_circularity, blob_min_convexity, blob_color
            )
            all_debug_data.update(debug_data_blob)
            method_used = "Blob Detector"
            parameters_used_info = f"Área mín: {area_minima}px | Min Circ: {blob_min_circularity:.2f} | Min Conv: {blob_min_convexity:.2f} | Cor: {'Preto' if blob_color == 0 else 'Branco'}"

        else:
            st.error(f"Método de análise desconhecido: {analysis_method}")
            return img_para_analise_bgr, 0, 0.0, all_debug_data, "Método Desconhecido", "N/A", analysis_date.strftime("%Y-%m-%d %H:%M:%S")

        end_time = time.time()
        processing_time = end_time - start_time

        st.success(f"Análise concluída ({method_used}) em {processing_time:.2f} segundos.")

        return resultado_final, contagem, processing_time, all_debug_data, method_used, parameters_used_info, analysis_date.strftime("%Y-%m-%d %H:%M:%S")

    except Exception as e:
        st.error(f"Erro durante a análise das colônias ({analysis_method}): {e}")
        method_used = analysis_method if analysis_method else "Erro"
        parameters_used = "Erro na análise"
        # Retorna a imagem BGR que entrou na análise (img_para_analise_bgr)
        return img_para_analise_bgr, 0, 0.0, all_debug_data, method_used, parameters_used, analysis_date.strftime("%Y-%m-%d %H:%M:%S")


# --- Inicialização do Estado da Sessão ---
if 'resultados_analise' not in st.session_state:
    st.session_state.resultados_analise = [] # Lista para guardar históricos
if 'imagem_atual' not in st.session_state:
    st.session_state.imagem_atual = None # Imagem OpenCV original carregada
if 'resultado_atual' not in st.session_state:
    st.session_state.resultado_atual = None # Dicionário com o último resultado
if 'nome_arquivo' not in st.session_state:
    st.session_state.nome_arquivo = "" # Nome do arquivo carregado

# --- Interface Streamlit ---

st.markdown("""
<style>
.navbar { background-color: #0E9046; padding: 10px; border-radius: 5px; text-align: center; color: black; font-size: 24px; font-weight: bold; margin-bottom: 20px; }
/* Add some space below metrics */
div[data-testid="stMetric"] {
    margin-bottom: 10px;
}
/* Style for history items */
.history-item {
    border: 1px solid #e0e0e0;
    border-radius: 5px;
    padding: 10px;
    margin-bottom: 10px;
}
</style>
<div class="navbar">Contador de Colônias BioMix 🦠
""", unsafe_allow_html=True)
# st.write("") # Removed for slightly less space

col_sidebar, col_main = st.columns([1, 3]) # Ajuste na proporção das colunas

with col_sidebar:
    st.markdown("### 📤 Upload e Opções")
    # st.markdown("---") # Optional separator
    uploaded_file = st.file_uploader("Carregar Imagem (.jpg, .png, .jpeg)", type=['jpg', 'png', 'jpeg'], key="fileuploader", label_visibility="collapsed")

    if st.session_state.get('imagem_atual') is not None:
         if st.button("🗑️ Limpar Imagem Atual", key="clear_button_sidebar", use_container_width=True):
            st.session_state.imagem_atual = None
            st.session_state.resultado_atual = None
            st.session_state.nome_arquivo = ""
            # st.query_params.clear() # Clear query params if you were using them
            st.rerun()
    st.markdown("---")

    st.markdown("### ⚙️ Ajustes de Análise")
    analysis_method = st.selectbox(
        "Método de Análise",
        ("Watershed (Picos Locais)", "Detector de Blobs"),
        key="analysis_method_selector",
        index=0 # Default to Watershed
    )
    # st.markdown("---") # Optional separator

    st.markdown("##### Parâmetros Comuns")
    area_minima_slider = st.slider("Área Mínima da Colônia (pixels)", 1, 500, 25, 1, key="area_slider")
    # st.markdown("---") # Optional separator

    st.markdown("##### Parâmetros Específicos")

    # Initialize parameters outside the conditional block
    peak_footprint_slider = None
    blob_min_circularity_val = 0.0 # Default
    blob_min_convexity_val = 0.0 # Default
    blob_color_val = 0 # Default (Preto)

    if analysis_method == "Watershed (Picos Locais)":
        peak_footprint_slider = st.slider("Tamanho Vizinhança Marcadores (px, ímpar)", 3, 51, 11, 2, key="peak_footprint_slider", help="Tamanho da área (ímpar) para encontrar centros (picos). Valores maiores separam colônias mais distantes.")
        # Set blob params to defaults when Watershed is selected (they won't be used but avoids errors)
        blob_min_circularity_val, blob_min_convexity_val, blob_color_val = 0.7, 0.8, 0
    elif analysis_method == "Detector de Blobs":
        blob_min_circularity_val = st.slider("Min Circularidade (0-1)", 0.0, 1.0, 0.7, 0.05, "%.2f", key="blob_min_circularity_slider", help="Filtra por quão circular a forma é (1.0 = círculo perfeito).")
        blob_min_convexity_val = st.slider("Min Convexidade (0-1)", 0.0, 1.0, 0.8, 0.05, "%.2f", key="blob_min_convexity_slider", help="Filtra por quão convexa a forma é (1.0 = sem 'buracos' ou reentrâncias).")
        blob_color_val = st.radio("Cor do Blob (Colônia)", [0, 255], index=0, format_func=lambda x: '⚫ Escuro' if x == 0 else '⚪ Claro', key="blob_color_radio", horizontal=True, help="Cor das colônias a serem detectadas (0=escuras/pretas, 255=claras/brancas).")
        # Set peak footprint to default when Blob is selected
        peak_footprint_slider = 11

    st.markdown("---")
    st.caption("BioMix Solutions © 2025")


with col_main:
    st.markdown("### 🔬 Análise da Imagem")
    col_img_original, col_img_resultado = st.columns(2)

    # --- Upload Logic ---
    if uploaded_file is not None:
        # Check if it's a new file upload
        if uploaded_file.name != st.session_state.get('nome_arquivo', ''):
            try:
                # Read image using PIL and convert to RGB immediately
                image_pil = Image.open(uploaded_file).convert('RGB')
                # Convert PIL (RGB) to OpenCV format (BGR) for processing
                img_original_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

                st.session_state.imagem_atual = img_original_cv
                st.session_state.nome_arquivo = uploaded_file.name
                st.session_state.resultado_atual = None # Clear previous result
                st.toast(f"Imagem '{uploaded_file.name}' carregada.", icon="✅")
                st.rerun() # Rerun to update displays immediately

            except Exception as e:
                st.error(f"Erro ao ler o arquivo de imagem: {e}")
                st.session_state.imagem_atual = None
                st.session_state.nome_arquivo = ""
                # No rerun here, let the error message show

    # --- Display Original Image / Input for Analysis ---
    with col_img_original:
        st.markdown("##### Imagem para Análise")
        if st.session_state.get('imagem_atual') is not None:
            img_display_orig = None
            caption_orig = f"Arquivo: {st.session_state.get('nome_arquivo', 'N/A')}"
            placa_detectada_info = ""

            # Decide which image to show: Prefer the one that went into analysis (post-crop)
            if st.session_state.get('resultado_atual') and st.session_state.resultado_atual.get('debug_data'):
                debug_data = st.session_state.resultado_atual['debug_data']
                if debug_data.get('imagem_para_analise') is not None:
                    img_display_orig = debug_data['imagem_para_analise'] # Already RGB
                    placa_detectada_info = "(Placa Detectada)" if debug_data.get('placa_detectada') else "(Placa Não Detectada / Recorte Falhou)"
                elif debug_data.get('recorte_tentativa') is not None: # Fallback to crop attempt
                     img_display_orig = debug_data['recorte_tentativa'] # Already RGB
                     placa_detectada_info = "(Placa Detectada)" if debug_data.get('placa_detectada') else "(Placa Não Detectada / Recorte Falhou)"

            # If no analysis done yet, show the initially loaded image (converted to RGB)
            if img_display_orig is None and st.session_state.get('imagem_atual') is not None:
                try:
                    # Original stored as BGR, convert to RGB for display
                    img_display_orig = cv2.cvtColor(st.session_state.imagem_atual, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    st.warning(f"Não foi possível converter a imagem original para exibição: {e}")

            # Display the chosen image
            if img_display_orig is not None:
                st.image(img_display_orig, caption=f"{caption_orig} {placa_detectada_info}", use_column_width='always')
            else:
                st.warning("Imagem original não disponível para exibição.")
        else:
            st.info("⬅️ Carregue uma imagem no painel esquerdo.")

    # --- Analysis Trigger and Result Display ---
    with col_img_resultado:
        st.markdown("##### Resultado da Análise")
        if st.session_state.get('imagem_atual') is not None:
            # Get current widget values safely
            current_area = area_minima_slider
            current_peak_footprint = peak_footprint_slider if analysis_method == "Watershed (Picos Locais)" else 11 # Use value or default
            current_blob_circ = blob_min_circularity_val if analysis_method == "Detector de Blobs" else 0.7 # Use value or default
            current_blob_conv = blob_min_convexity_val if analysis_method == "Detector de Blobs" else 0.8 # Use value or default
            current_blob_color = blob_color_val if analysis_method == "Detector de Blobs" else 0 # Use value or default

            if st.button(f"▶️ Analisar com {analysis_method}", key="analyze_button_main", use_container_width=True, type="primary"):
                with st.spinner(f'Analisando com {analysis_method}... Por favor, aguarde.'):
                    resultado_final_cv, contagem, tempo, debug_data, method_used, parameters_used_info, analysis_date_str = analisar_colonias(
                        st.session_state.imagem_atual, # Pass the original BGR image
                        current_area,
                        current_peak_footprint,
                        analysis_method,
                        current_blob_circ,
                        current_blob_conv,
                        current_blob_color
                    )

                    # Store results in session state
                    st.session_state.resultado_atual = {
                        'imagem': resultado_final_cv, # Result image (should be BGR)
                        'contagem': contagem,
                        'tempo': tempo,
                        'nome': st.session_state.nome_arquivo,
                        'area_minima': current_area,
                        'peak_footprint': current_peak_footprint if method_used == "Watershed" else None,
                        'blob_min_circularity': current_blob_circ if method_used == "Blob Detector" else None,
                        'blob_min_convexity': current_blob_conv if method_used == "Blob Detector" else None,
                        'blob_color': current_blob_color if method_used == "Blob Detector" else None,
                        'debug_data': debug_data, # Dictionary of debug images/data
                        'method': method_used,
                        'parameters_info': parameters_used_info,
                        'analysis_date': analysis_date_str # Store analysis date as string
                    }

                    # Add to history only if successful and not an error placeholder
                    if "Erro" not in method_used and method_used != "Método Desconhecido":
                        MAX_HISTORICO = 15 # Increase history size if needed
                        # Create a deep copy for the history list
                        hist_entry = st.session_state.resultado_atual.copy()
                        # Ensure debug data is also copied properly if it contains mutable types (like numpy arrays)
                        hist_entry['debug_data'] = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in debug_data.items()}

                        st.session_state.resultados_analise.insert(0, hist_entry)
                        st.session_state.resultados_analise = st.session_state.resultados_analise[:MAX_HISTORICO]

                    st.rerun() # Rerun to display the new result and update history

            # Display the latest analysis result
            if st.session_state.get('resultado_atual'):
                res_atual = st.session_state.resultado_atual
                if 'imagem' in res_atual and isinstance(res_atual['imagem'], np.ndarray):
                    try:
                        # Result image is stored as BGR, convert to RGB for display
                        img_resultado_rgb = cv2.cvtColor(res_atual['imagem'], cv2.COLOR_BGR2RGB)
                        st.image(img_resultado_rgb,
                                 caption=f"Resultado ({res_atual.get('method', 'N/A')}): {res_atual.get('contagem', 0)} colônias",
                                 use_column_width='always')
                    except Exception as e_res:
                        st.warning(f"Não foi possível exibir a imagem de resultado: {e_res}")
                        st.write("Detalhes do Resultado:")
                        st.json(res_atual, expanded=False) # Show JSON if image fails

                elif "Erro" in res_atual.get('method', ''):
                     st.error(f"A última análise falhou. Método: {res_atual.get('method', 'N/A')}. Verifique os parâmetros ou a imagem.")
                else:
                    # Should not happen if analysis ran correctly, but good fallback
                    st.warning("Resultado da análise anterior inválido ou imagem não encontrada.")
            elif st.session_state.get('imagem_atual') is not None:
                 st.info("Clique no botão ▶️ Analisar para ver o resultado.")


    # --- Informações da Análise Atual (Displayed below images) ---
    if st.session_state.get('resultado_atual') and "Erro" not in st.session_state.resultado_atual.get('method', "Erro"):
        st.markdown("---")
        st.markdown("#### 📊 Informações da Análise Atual")
        res_atual = st.session_state.resultado_atual
        info_col1, info_col2 = st.columns(2)

        # Truncate long filenames for display
        nome_display = res_atual.get('nome', 'N/A')
        if not nome_display: nome_display = "Indisponível"
        max_len = 25
        if len(nome_display) > max_len: nome_display = nome_display[:max_len-3] + "..."

        with info_col1:
            st.metric("Arquivo", nome_display)
            st.metric("Método Usado", res_atual.get('method', 'N/A'))
        with info_col2:
            st.metric("Colônias Detectadas", f"{res_atual.get('contagem', 0):,}")
            st.metric("Tempo Análise (s)", f"{res_atual.get('tempo', 0.0):.3f}")

        st.caption(f"Parâmetros: {res_atual.get('parameters_info', 'N/A')}")
        st.caption(f"Analisado em: {res_atual.get('analysis_date', 'N/A')}")


    # --- Área de Debug (Expander) ---
    st.markdown("---")
    with st.expander("🔍 Ver Etapas Intermediárias (Debug)"):
        if not st.session_state.get('resultado_atual') or not st.session_state.resultado_atual.get('debug_data'):
            st.info("Realize uma análise para ver as etapas de depuração.")
        else:
            debug_images_current = st.session_state.resultado_atual['debug_data']
            method_used_for_debug = st.session_state.resultado_atual.get('method', 'N/A')

            # Define the desired order of debug steps
            debug_order = ['imagem_original', 'recorte_tentativa', 'placa_detectada', 'imagem_para_analise']
            if method_used_for_debug == "Watershed":
                debug_order.extend(['imagem_entrada_watershed', 'blur_watershed', 'binarizada', 'morfologia_opening', 'dist_transform', 'picos_locais_marcadores', 'regiao_desconhecida', 'watershed_labels_colored'])
            elif method_used_for_debug == "Blob Detector":
                debug_order.extend(['imagem_entrada_blob', 'blur_blob', 'binarizada_blob', 'keypoints_detectados'])

            keys_to_show = [k for k in debug_order if k in debug_images_current]

            if not keys_to_show:
                st.warning("Nenhuma imagem ou dado de debug foi gerado ou encontrado para esta análise.")
            else:
                num_cols_debug = 3 # Adjust number of columns for debug images
                debug_cols = st.columns(num_cols_debug)
                col_idx = 0
                for key in keys_to_show:
                    debug_item = debug_images_current[key]
                    with debug_cols[col_idx % num_cols_debug]:
                        # Create a more descriptive title
                        caption = f"{key.replace('_', ' ').replace('blob', 'Blob').replace('watershed', 'Watershed').title()}"
                        try:
                            if isinstance(debug_item, np.ndarray): # It's an image
                                # Assume images in debug_data are already RGB or Grayscale suitable for display
                                st.image(debug_item, caption=caption, use_column_width='always')
                            elif isinstance(debug_item, bool) and key == 'placa_detectada': # Special case for boolean
                                st.metric("Placa Detectada?", "✔️ Sim" if debug_item else "❌ Não")
                            elif isinstance(debug_item, (int, float, str, bool)): # Display other simple data types
                                 st.metric(caption, str(debug_item))
                            # Add more types if needed
                        except Exception as e_dbg:
                            st.warning(f"Erro ao exibir debug '{key}': {e_dbg}", icon="⚠️")

                    col_idx += 1

    # --- Histórico e Download ---
    if st.session_state.get('resultados_analise'):
        st.divider()
        st.markdown("### 📚 Histórico e Download das Análises")

        # Dictionary to store selection state (index -> bool)
        selected_indices = {}

        MAX_DISPLAY_HISTORICO = 10 # Show max 10 items directly
        results_to_display = st.session_state.resultados_analise[:MAX_DISPLAY_HISTORICO]

        for i, res in enumerate(results_to_display):
             with st.container(border=True): # Use container for better visual separation
                cols_hist = st.columns([0.5, 1.5, 3, 1.5, 0.5]) # Adjusted column widths

                with cols_hist[0]: # Checkbox
                     # Use index 'i' as part of the key
                     selected = st.checkbox("", key=f"select_hist_{i}", value=False, label_visibility="collapsed")
                     selected_indices[i] = selected # Store selection state

                with cols_hist[1]: # Thumbnail
                    if 'imagem' in res and isinstance(res['imagem'], np.ndarray):
                        try:
                            thumb_rgb = cv2.cvtColor(res['imagem'], cv2.COLOR_BGR2RGB)
                            st.image(thumb_rgb, width=100, caption=f"#{i+1}")
                        except Exception as e_thumb:
                            st.caption(f"#{i+1} (Thumb err: {e_thumb})")
                    else:
                        st.caption(f"#{i+1} (No Thumb)")

                with cols_hist[2]: # Info
                    nome_hist = res.get('nome', 'N/A')
                    if not nome_hist: nome_hist = "Indisponível"
                    max_len_hist = 30
                    if len(nome_hist) > max_len_hist: nome_hist = nome_hist[:max_len_hist-3] + "..."
                    st.markdown(f"**{nome_hist}**")
                    st.caption(f"Método: {res.get('method', 'N/A')} | Colônias: **{res.get('contagem', 'N/A')}**")
                    st.caption(f"Params: {res.get('parameters_info', 'N/A')}")

                with cols_hist[3]: # Date & Time
                     st.caption(f"Data: {res.get('analysis_date', 'N/A')}")
                     st.caption(f"Tempo: {res.get('tempo', 0.0):.3f} s")

                with cols_hist[4]: # Delete Button
                     # Use index 'i' as part of the key
                     if st.button("🗑️", key=f"delete_hist_{i}", help="Excluir este item do histórico"):
                         del st.session_state.resultados_analise[i]
                         st.toast(f"Item #{i+1} removido do histórico.", icon="🗑️")
                         st.rerun() # Rerun immediately to update list and prevent index errors

        st.markdown("---") # Separator before download button

        # --- Download Button and Logic ---
        if any(selected_indices.values()): # Show button only if at least one item is selected
             st.markdown("##### Download Selecionados")
             col_btn1, col_btn2 = st.columns(2)

             with col_btn1:
                 if st.button("💾 Download (.zip + .xlsx)", key="download_selected", use_container_width=True):
                    selected_items_data = []
                    files_for_zip = [] # List of tuples: (filename_in_zip, image_bytes)

                    st.toast("Preparando arquivos para download...", icon="⏳")
                    progress_bar = st.progress(0, text="Coletando dados...")

                    num_selected = sum(selected_indices.values())
                    items_processed = 0

                    for idx, is_selected in selected_indices.items():
                        if is_selected and idx < len(st.session_state.resultados_analise):
                            res = st.session_state.resultados_analise[idx]
                            base_filename = os.path.splitext(res.get('nome', f'analise_{idx}'))[0] # Remove extension
                            # Sanitize filename for ZIP
                            safe_base_filename = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in base_filename)
                            prefix = f"Analise_{idx+1}_{safe_base_filename}"

                            # 1. Prepare Data for Table
                            table_row = {
                                "ID": idx + 1,
                                "Arquivo": res.get('nome', 'N/A'),
                                "Contagem": res.get('contagem', 'N/A'),
                                "Metodo": res.get('method', 'N/A'),
                                "Area Min (px)": res.get('area_minima', 'N/A'),
                                "Pico Viz (px)": res.get('peak_footprint', 'N/A') if res.get('method') == "Watershed" else "-",
                                "Min Circ": f"{res.get('blob_min_circularity', 'N/A'):.2f}" if isinstance(res.get('blob_min_circularity'), float) else "-",
                                "Min Conv": f"{res.get('blob_min_convexity', 'N/A'):.2f}" if isinstance(res.get('blob_min_convexity'), float) else "-",
                                "Cor Blob": ('Preto' if res.get('blob_color') == 0 else 'Branco') if res.get('method') == "Blob Detector" and res.get('blob_color') is not None else "-",
                                "Tempo (s)": f"{res.get('tempo', 0.0):.3f}",
                                "Data Analise": res.get('analysis_date', 'N/A')
                            }
                            selected_items_data.append(table_row)

                            # 2. Prepare Result Image for Zip
                            if 'imagem' in res and isinstance(res['imagem'], np.ndarray):
                                img_bytes = opencv_img_to_bytes(res['imagem'], format='PNG')
                                if img_bytes:
                                    files_for_zip.append((f"{prefix}_Resultado.png", img_bytes))

                            # 3. Prepare Debug Images for Zip
                            if 'debug_data' in res and isinstance(res['debug_data'], dict):
                                for key, dbg_img in res['debug_data'].items():
                                    if isinstance(dbg_img, np.ndarray): # Check if it's an image
                                        dbg_bytes = opencv_img_to_bytes(dbg_img, format='PNG')
                                        if dbg_bytes:
                                             # Sanitize key for filename
                                             safe_key = "".join(c if c.isalnum() else '_' for c in key)
                                             files_for_zip.append((f"{prefix}_Debug_{safe_key}.png", dbg_bytes))

                            items_processed += 1
                            progress_bar.progress(items_processed / num_selected, text=f"Processando item {idx+1}...")


                    if not selected_items_data:
                        st.warning("Nenhum item válido selecionado para download.", icon="⚠️")
                        progress_bar.empty()
                    else:
                        try:
                            # Create Excel
                            progress_bar.progress(1.0, text="Gerando Excel...")
                            df = pd.DataFrame(selected_items_data)
                            excel_buffer = io.BytesIO()
                            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                                df.to_excel(writer, index=False, sheet_name='Resumo Analises')
                                # Auto-adjust columns (optional, might slow down for many columns)
                                # for column in df:
                                #     column_width = max(df[column].astype(str).map(len).max(), len(column))
                                #     col_idx = df.columns.get_loc(column)
                                #     writer.sheets['Resumo Analises'].set_column(col_idx, col_idx, column_width)
                            excel_data = excel_buffer.getvalue()

                            # Create Zip
                            progress_bar.progress(1.0, text="Gerando ZIP...")
                            zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                # Add Excel to Zip
                                zip_file.writestr("Resumo_Analises_BioMix.xlsx", excel_data)
                                # Add Images to Zip
                                for fname, fdata in files_for_zip:
                                    zip_file.writestr(fname, fdata)
                            zip_data = zip_buffer.getvalue()

                            progress_bar.empty() # Remove progress bar
                            st.toast("Arquivos prontos para download!", icon="🎉")

                            # Provide download buttons side-by-side using columns again if desired, or stack them
                            st.download_button(
                                label="⬇️ Baixar Tabela Resumo (.xlsx)",
                                data=excel_data,
                                file_name="Resumo_Analises_BioMix.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="download_excel_btn",
                                use_container_width=True
                            )
                            st.download_button(
                                label="🖼️ Baixar Imagens + Tabela (.zip)",
                                data=zip_data,
                                file_name="Analises_BioMix_Resultados.zip",
                                mime="application/zip",
                                key="download_zip_btn",
                                use_container_width=True
                            )

                        except Exception as e_dl:
                            st.error(f"Erro ao gerar arquivos para download: {e_dl}")
                            progress_bar.empty()
        else:
            st.caption("Selecione um ou mais itens no histórico para habilitar o download.")

# Final check - if no image is loaded, maybe display a welcome message or instructions
if st.session_state.get('imagem_atual') is None and not uploaded_file :
     # Check if uploaded_file exists to avoid showing this after upload error
     with col_main:
         st.info("Bem-vindo ao Contador de Colônias BioMix! Carregue uma imagem para começar.")

# Dividindo o espaço para posicionar o botão corretamente
_, col_action2, _ = st.columns([4, 1, 1])

# with col_action2:
#         # Botão antigo para limpar todo o histórico
#     if st.button("Limpar TODO o Histórico", key="clear_history_button_bottom", type="secondary"):
#             st.session_state.resultados_analise = []
#             st.toast("Histórico de análises limpo.")
#             time.sleep(0.5)
#             st.rerun()

    # st.markdown("---")
    # st.markdown("#### Resumo do Histórico (Tabela)")
    # if st.session_state.resultados_analise:
    #     df_resultados = pd.DataFrame([{
    #         'Imagem': r.get('nome', 'N/A')[:30] + '...' if len(r.get('nome', '')) > 30 else r.get('nome', 'N/A'),
    #         'Colônias': r.get('contagem', 0),
    #         'Tempo (s)': f"{r.get('tempo', 0.0):.2f}",
    #         'Área Mínima': r.get('area_minima', 'N/A'),
    #         'Vizinhança Marcadores': r.get('peak_footprint', 'N/A')
    #     } for r in st.session_state.resultados_analise]) # Itera sobre o estado atual do histórico

# =========================
# Seção: Resumo do Histórico (Tabela)
# =========================

st.markdown("## 📊 Resumo do Histórico")

# Exibir histórico se houver dados
if st.session_state.resultados_analise:
    # Criar o DataFrame completo a partir dos dados do histórico
    df_resultados_completo = pd.DataFrame(st.session_state.resultados_analise)

    # --- Preparar DataFrame APENAS para exibição na tabela Streamlit ---
    # Selecionar apenas as colunas desejadas para o resumo
    colunas_resumo = ['nome', 'contagem', 'tempo', 'analysis_date', 'method'] # Incluí 'method' para contexto, opcional

    # Garantir que as colunas existam no DataFrame antes de selecionar
    colunas_resumo_existentes = [col for col in colunas_resumo if col in df_resultados_completo.columns]

    df_resultados_para_tabela = df_resultados_completo[colunas_resumo_existentes].copy() # Use .copy() para evitar SettingWithCopyWarning

    # Opcional: Renomear colunas para melhor apresentação na tabela
    nomes_colunas_amigaveis = {
        'nome': 'Arquivo',
        'contagem': 'Colônias',
        'tempo': 'Tempo (s)',
        'analysis_date': 'Data/Hora Análise',
        'method': 'Método' # Renomeia a coluna method se incluída
    }
    df_resultados_para_tabela = df_resultados_para_tabela.rename(columns=nomes_colunas_amigaveis)

    # Centralizar e expandir a tabela
    col1, col2, col3 = st.columns([0.1, 5, 0.1])
    with col2:
        # Exibir a tabela resumo
        try:
            # Não precisamos nos preocupar com arrays de imagem ou debug aqui, pois removemos essas colunas
            # No entanto, se houver outros tipos complexos nas colunas selecionadas, esta verificação ainda é útil:
            for col in df_resultados_para_tabela.columns:
                 # Verifica se a coluna ainda contém listas, dicionários ou arrays (improvável após seleção, mas seguro)
                if df_resultados_para_tabela[col].apply(lambda x: isinstance(x, (list, dict, np.ndarray))).any():
                    df_resultados_para_tabela[col] = df_resultados_para_tabela[col].apply(lambda x: str(x))


            st.dataframe(df_resultados_para_tabela, use_container_width=True)
        except Exception as e:
            st.error(f"Erro ao exibir a tabela resumo: {e}")
            # Opcional: st.write("Dados do DataFrame para debug:", df_resultados_para_tabela)


    # Botão centralizado abaixo da tabela
    # Este botão já está bem posicionado, não precisa estar dentro do `with col2:`
    # Vamos movê-lo para fora para garantir que ele apareça no layout correto.
    # _, col_btn, _ = st.columns([4, 2, 4])
    # with col_btn: # REMOVER este 'with' e sua indentação

    # Botão Limpar TODO o Histórico (Mantido centralizado abaixo da tabela resumo)
    _, col_limpar, _ = st.columns([4, 2, 4])
    with col_limpar:
        if st.button("🧹 Limpar TODO o Histórico", key="clear_history_button_bottom_resumo", type="secondary", use_container_width=True):
            st.session_state.resultados_analise = []
            st.toast("Histórico de análises limpo.")
            st.rerun()

# Este 'else' pertence ao 'if st.session_state.resultados_analise:' e exibe a mensagem quando o histórico está vazio
else:
     st.info("Nenhuma análise realizada ainda.")

# As seções de download e histórico individual (com checkboxes)
# devem permanecer inalteradas e continuar usando st.session_state.resultados_analise
# para acessar os dados completos, incluindo as imagens.


