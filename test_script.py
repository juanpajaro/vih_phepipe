# test_script.py
import os
import time

print("Iniciando tarea de prueba...")
print(f"Ejecutando en nodo: {os.uname().nodename}")
print(f"PID del proceso: {os.getpid()}")

# Simulamos una tarea
time.sleep(10)

print("¡Tarea completada exitosamente!")
print("Fin del script.")