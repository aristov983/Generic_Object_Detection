import customtkinter as ctk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import pandas as pd
from datetime import datetime
import os, sys, webbrowser

VERSION = 'version 1.0'

def buscar_camaras(max_index=5):
    disponibles = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            disponibles.append(i)
            cap.release()
    return disponibles

# Selección de cámara antes de iniciar la app principal
root_select = ctk.CTk()
root_select.title("Seleccionar cámara")
root_select.geometry("300x150")
root_select.resizable(False, False)

camaras = buscar_camaras()
if not camaras:
    messagebox.showerror("Error", "No se detectaron cámaras.")
    root_select.destroy()
    exit()

selected_index = ctk.IntVar(value=camaras[0])

label = ctk.CTkLabel(root_select, text="Selecciona la cámara a usar:")
label.pack(pady=10)

option_menu = ctk.CTkOptionMenu(root_select, values=[str(i) for i in camaras], variable=selected_index)
option_menu.pack(pady=10)

def confirmar():
    root_select.destroy()

boton = ctk.CTkButton(root_select, text="Confirmar", command=confirmar)
boton.pack(pady=10)

root_select.mainloop()

CAM_INDEX = int(selected_index.get())

def ruta_de_recursos(relative_path):
    try:
        base_path = sys._MEIPASS  # Usado por PyInstaller
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Variables globales
carpeta_seleccionada = None
detecciones = []
ultima_deteccion = datetime.min  # Al inicio del archivo, después de tus variables globales

# Cargar modelo YOLOv8
model = YOLO(ruta_de_recursos(r"Model\best.pt"))  # Cambia la ruta si es necesario

# Inicializar cámara con el índice seleccionado
cap = cv2.VideoCapture(CAM_INDEX)

# --- Funciones de interfaz ---

def seleccionar_carpeta():
    global carpeta_seleccionada
    carpeta = filedialog.askdirectory()
    if carpeta:
        carpeta_seleccionada = carpeta
        terminal_text.insert("end", f"Carpeta seleccionada: {carpeta_seleccionada}\n")
        terminal_text.see("end")

def generar_excel():
    if not carpeta_seleccionada:
        messagebox.showerror("Error", "Debe seleccionar carpeta primero")
        terminal_text.insert("end", "Error: Debe seleccionar carpeta primero\n")
        terminal_text.see("end")
        return

    terminal_text.insert("end", f"Detecciones actuales: {detecciones}\n")

    if not detecciones:
        messagebox.showwarning("Advertencia", "No hay detecciones para exportar.")
        terminal_text.insert("end", "Advertencia: No hay detecciones para exportar.\n")
        terminal_text.see("end")
        return

    # Crear DataFrame con nombre y hora
    df = pd.DataFrame(detecciones, columns=["Marca", "Hora"])
    ruta_excel = os.path.join(carpeta_seleccionada, "object_detections.xlsx")
    try:
        df.to_excel(ruta_excel, index=False)
        messagebox.showinfo("Éxito", f"Excel generado en:\n{ruta_excel}")
        terminal_text.insert("end", f"Excel generado en: {ruta_excel}\n")
        terminal_text.see("end")
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo generar el Excel:\n{e}")
        terminal_text.insert("end", f"Error al generar Excel: {e}\n")
        terminal_text.see("end")

def mostrar_camara():
    global ultima_deteccion
    ret, frame = cap.read()
    if ret:
        results = model(frame)[0]
        annotated_frame = results.plot()

        # Filtrar detecciones por confianza > 0.8
        nombres_detectados = []
        ahora = datetime.now()
        for cls, conf in zip(results.boxes.cls, results.boxes.conf):
            if conf > 0.8:
                nombre = model.names[int(cls)]
                nombres_detectados.append(nombre)

        if nombres_detectados and (ahora - ultima_deteccion).total_seconds() >= 2:
            for nombre in nombres_detectados:
                detecciones.append((nombre, ahora.strftime("%H:%M:%S")))  # Guarda nombre y hora
            ultima_deteccion = ahora
            # Solo guardar las últimas 100 detecciones
            if len(detecciones) > 100:
                del detecciones[:-100]

        # Actualizar historial (últimas 20)
        ultimas = detecciones[-20:]
        history_text.configure(state='normal')
        history_text.delete("0.0", "end")
        for d in ultimas:
            history_text.insert("end", f"Se detectó: {d[0]} a las {d[1]}\n")
        history_text.configure(state='disabled')

        # Mostrar imagen en el label
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((640, 480))
        imgtk = ImageTk.PhotoImage(image=img)
        camara_label.imgtk = imgtk
        camara_label.configure(image=imgtk)

    camara_label.after(30, mostrar_camara)

def on_closing():
    cap.release()
    root.destroy()

# --- Interfaz gráfica ---

root = ctk.CTk()
root.title(f'Detector Botellas {VERSION}')
root.geometry("1280x720")
root.columnconfigure((0,1,2,3,4,5,6), weight=1)
root.rowconfigure((0,1,2), weight=1)
root.iconbitmap(ruta_de_recursos(r"Assets\Image and icon\Program_Icon.ico"))

# HISTORIAL
frame_history = ctk.CTkFrame(root)
frame_history.grid(row=0, column=0, rowspan=5, sticky="nswe", padx=20, pady=20)
frame_history.columnconfigure(0, weight=1)
frame_history.rowconfigure(0, weight=1)
frame_history.rowconfigure(1, weight=6)
frame_history.grid_propagate(False)

history_title = ctk.CTkLabel(frame_history, text="Historial", font=("Arial", 16, "bold"), anchor="center")
history_title.grid(row=0, column=0, sticky="n", padx=80, pady=20)

history_text = ctk.CTkTextbox(frame_history, width=300, height=600, font=("Arial", 12))
history_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
history_text.configure(state='disabled')

# CÁMARA
frame_camara = ctk.CTkFrame(root)
frame_camara.grid(row=0, column=1, rowspan=4, columnspan=4, sticky="nsew", padx=20, pady=20)
frame_camara.columnconfigure(0, weight=1)
frame_camara.rowconfigure(0, weight=1)
frame_camara.grid_propagate(False)


camara_label = ctk.CTkLabel(frame_camara, text="")
camara_label.grid(row=0, column=0, sticky="nsew")

# TERMINAL
frame_terminal = ctk.CTkFrame(root)
frame_terminal.grid(row=4, column=1, columnspan=6, sticky="nsew", padx=20, pady=10)
frame_terminal.columnconfigure(0, weight=1)
frame_terminal.rowconfigure(0, weight=1)
frame_terminal.grid_propagate(False)

terminal_text = ctk.CTkTextbox(frame_terminal, width=900, height=100, font=("Arial", 12))
terminal_text.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

# BOTONES
frame_botones = ctk.CTkFrame(root)
frame_botones.grid(row=0, column=6, rowspan=4, sticky="nsew", padx=20, pady=20)
frame_botones.columnconfigure(0, weight=1)
frame_botones.rowconfigure((0,1,2), weight=1)
frame_botones.grid_propagate(False)

#Logo greenpath
logo_greenpath=Image.open(ruta_de_recursos(r"Assets\Image and icon\Logo.png")) #Carga la imagen del logo
logo=ctk.CTkImage(light_image=logo_greenpath, dark_image=logo_greenpath, size=(180, 100))
logo_label=ctk.CTkLabel(frame_botones, image=logo, text="")
logo_label.grid(row=0, column=0, columnspan=2, sticky="nsew")

boton_seleccionarcarpeta = ctk.CTkButton(frame_botones, text="Seleccionar Carpeta", command=seleccionar_carpeta)
boton_seleccionarcarpeta.grid(row=1, column=0, sticky="nsew", padx=20, pady=20)

boton_generarexcel = ctk.CTkButton(frame_botones, text="Generar Excel", command=generar_excel)
boton_generarexcel.grid(row=2, column=0, sticky="nsew", padx=20, pady=20)

frame_footer = ctk.CTkFrame(root, fg_color="transparent")
frame_footer.grid(row=4, column=0, sticky="sw", padx=25, pady=5)
frame_footer.columnconfigure(0, weight=1)
footer_label = ctk.CTkLabel(
    frame_footer,
    text="Aristov983, NicolasDiazAg, Joaco2801 © 2025",
    font=("Arial", 10)
)
footer_label.grid(row=0, column=0, sticky="w")

def abrir_github():
    webbrowser.open_new("https://github.com/aristov983/Generic_Object_Detection")

github_link = ctk.CTkButton(
    frame_footer,
    text="GitHub",
    font=("Arial", 10, "underline"),
    fg_color="transparent",
    hover_color="#e0e0e0",
    text_color="#1a73e8",
    command=abrir_github,
    width=40,
    height=20
)
github_link.grid(row=0, column=1, sticky="w", padx=(10,0))


# Iniciar cámara y detección
mostrar_camara()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()