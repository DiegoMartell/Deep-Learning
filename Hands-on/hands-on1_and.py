# Perceptron para reconocer la compuerta logica AND


def step(v):
    if v >= 0:
        return 1
    return 0


def perceptron(x, t, max_it=1000):

    f = False
    epoca = 0

    w1 = 0.1
    w2 = 0.1
    b = 0.1

    while not f and epoca < max_it:
        f = True
        print(f"\n------- EPOCA {epoca + 1} -------")

        for k in range(len(x)):
            x1 = x[k][0]
            x2 = x[k][1]
            esperado = t[k]

            # Suma ponderada
            v = (x1 * w1) + (x2 * w2) + b

            y = step(v)

            # Error
            e = esperado - y

            print(f"\nPatron: [{x1}, {x2}]")
            print(f"Clase esperada: {esperado}")

            print("Suma ponderada:")
            print(f"v = ({x1} * {w1:.2f}) + ({x2} * {w2:.2f}) + {b:.2f}")
            print(f"v = {v:.2f}")

            print("Funcion de activacion:")
            print(f"y = step({v:.2f})")
            print(f"y = {y}")

            print("Error:")
            print(f"e = {esperado} - {y} = {e}")

            # Actualizacion de pesos y bias
            if e != 0:
                w1 = w1 + (e * x1)
                w2 = w2 + (e * x2)
                b = b + e
                f = False

                print("Se actualizan los parametros:")
                print(f"w1 = {w1:.2f}")
                print(f"w2 = {w2:.2f}")
                print(f"b  = {b:.2f}")
            else:
                print("No se actualizan los parametros.")

        epoca += 1

    print("\n----------------------------------------")
    print("ENTRENAMIENTO TERMINADO")
    print(f"Epocas utilizadas: {epoca}")
    print("\nPesos optimos:")
    print(f"w1 = {w1:.2f}")
    print(f"w2 = {w2:.2f}")
    print(f"b  = {b:.2f}")

    return w1, w2, b

x = [[0, 0], [0, 1], [1, 0], [1, 1]]

t = [0, 0, 0, 1]

# Entrenamiento
w1, w2, b = perceptron(x, t)

# Comprobacion final
print("\n----------------------------------------")
print("COMPROBACION FINAL")

for k in range(len(x)):
    x1 = x[k][0]
    x2 = x[k][1]
    esperado = t[k]

    v = (x1 * w1) + (x2 * w2) + b
    y = step(v)

    print(f"\nPatron: [{x1}, {x2}]")
    print(
        f"Suma ponderada: {x1} * {w1:.2f} + {x2} * {w2:.2f} + {b:.2f} = {v:.2f}"
    )
    print(f"step({v:.2f}) = {y}")
    print(f"Esperado: {esperado} | Obtenido: {y}")

    if esperado == y:
        print("Resultado: CORRECTO")
    else:
        print("Resultado: INCORRECTO")