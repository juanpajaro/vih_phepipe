#!/usr/bin/env python3
import sys

arg = sys.argv[1]
print(arg[::-1])

lista_capturada = sys.argv[2].split(",")
print(lista_capturada)
print(type(lista_capturada))
#print(lista_capturada[0])
for i in range(len(lista_capturada)):
    print(lista_capturada[i])