# Perceptron Orientado a Objetos para la compuerta logica OR

class Perceptron:

    def __init__(self, eta=0.1, max_it=1000):
        self.eta = eta
        self.max_it = max_it

        self.w1 = 0.1
        self.w2 = 0.1
        self.b = 0.1
        self.epoca = 0

    def step(self, v):
        if v >= 0:
            return 1
        return 0

    def entrenar(self, x, t):
        f = False
        self.epoca = 0

        while not f and self.epoca < self.max_it:
            f = True
            print(f"\n------- EPOCA {self.epoca + 1} -------")

            for k in range(len(x)):
                x1 = x[k][0]
                x2 = x[k][1]
                esperado = t[k]

                # Suma ponderada
                v = (x1 * self.w1) + (x2 * self.w2) + self.b

                y = self.step(v)

                # Error
                e = esperado - y

                print(f"\nPatron: [{x1}, {x2}]")
                print(f"Clase esperada: {esperado}")

                print("Suma ponderada:")
                print(
                    f"v = ({x1} * {self.w1:.2f}) + "
                    f"({x2} * {self.w2:.2f}) + {self.b:.2f}"
                )
                print(f"v = {v:.2f}")

                print("Funcion de activacion:")
                print(f"y = step({v:.2f})")
                print(f"y = {y}")

                print("Error:")
                print(f"e = {esperado} - {y} = {e}")

                # Actualizacion de pesos y bias
                if e != 0:
                    self.w1 = self.w1 + (self.eta * e * x1)
                    self.w2 = self.w2 + (self.eta * e * x2)
                    self.b = self.b + (self.eta * e)
                    f = False

                    print("Se actualizan los parametros:")
                    print(f"w1 = {self.w1:.2f}")
                    print(f"w2 = {self.w2:.2f}")
                    print(f"b  = {self.b:.2f}")
                else:
                    print("No se actualizan los parametros.")

            self.epoca += 1

        print("\n----------------------------------------")
        print("ENTRENAMIENTO TERMINADO")
        print(f"Epocas utilizadas: {self.epoca}")
        print("\nPesos optimos:")
        print(f"w1 = {self.w1:.2f}")
        print(f"w2 = {self.w2:.2f}")
        print(f"b  = {self.b:.2f}")

        return self.w1, self.w2, self.b

    def comprobar(self, x, t):
        print("\n----------------------------------------")
        print("COMPROBACION FINAL")

        for k in range(len(x)):
            x1 = x[k][0]
            x2 = x[k][1]
            esperado = t[k]

            v = (x1 * self.w1) + (x2 * self.w2) + self.b
            y = self.step(v)

            print(f"\nPatron: [{x1}, {x2}]")
            print(
                f"Suma ponderada: {x1} * {self.w1:.2f} + "
                f"{x2} * {self.w2:.2f} + {self.b:.2f} = {v:.2f}"
            )
            print(f"step({v:.2f}) = {y}")
            print(f"Esperado: {esperado} | Obtenido: {y}")

            if esperado == y:
                print("Resultado: CORRECTO")
            else:
                print("Resultado: INCORRECTO")


x = [[0, 0], [0, 1], [1, 0], [1, 1]]

t = [0, 1, 1, 1]

p = Perceptron(eta=0.1)

# Entrenamiento
w1, w2, b = p.entrenar(x, t)

# Comprobacion final
p.comprobar(x, t)