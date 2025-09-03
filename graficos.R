library(ggplot2)

#ESCENARIO CONTROLADO

# Datos
precision <- data.frame(
  Participante = paste("P", 1:25, sep=""),
  Precision = c(100,100,100,100,100,99.07,98.67,98.52,97.5,97.5,
                97.3,97.03,96.36,96.33,95.54,94.52,93.83,93,93,
                92.78,90.1,89.69,89.68,89.47,77.46)
)

# Ordenar por precisión
precision$Participante <- factor(precision$Participante,
                                 levels = precision$Participante[order(precision$Precision)])

# Dot plot
ggplot(precision, aes(x = Participante, y = Precision)) +
  geom_point(size = 3, color = "steelblue") +
  geom_hline(yintercept = c(100, 90, 80), linetype = "dashed", color = "red") +
  coord_flip() +
  labs(title = "Precisión por Participante",
       x = "Participante", y = "Precisión (%)") +
  theme_minimal()



# violin plot

ggplot(precision, aes(x = "", y = Precision)) +
  geom_violin(fill = "lightblue", color = "black") +
  geom_jitter(width = 0.1, size = 2, color = "darkblue") +
  labs(title = "Distribución de Precisión por Participante",
       x = "", y = "Precisión (%)") +
  theme_minimal()





#escenarios no controlados
# Datos
precision <- data.frame(
  Participante = paste("P", 1:25, sep=""),
  Precision = c(96.77,
                95.24,
                90.28,
                90.27,
                89.74,
                89.7,
                86.87,
                86.13,
                84.1,
                83.83,
                83.67,
                83.67,
                82.5,
                81.48,
                80.82,
                80.5,
                77.08,
                75.63,
                73.77,
                73.33,
                73.12,
                71.7,
                69.14,
                67.35,
                66.67)
)

# Ordenar por precisión
precision$Participante <- factor(precision$Participante,
                                 levels = precision$Participante[order(precision$Precision)])

# Dot plot
ggplot(precision, aes(x = Participante, y = Precision)) +
  geom_point(size = 3, color = "steelblue") +
  geom_hline(yintercept = c(100, 90, 80), linetype = "dashed", color = "red") +
  coord_flip() +
  labs(title = "Precisión por Participante",
       x = "Participante", y = "Precisión (%)") +
  theme_minimal()



# violin plot

ggplot(precision, aes(x = "", y = Precision)) +
  geom_violin(fill = "lightblue", color = "black") +
  geom_jitter(width = 0.1, size = 2, color = "darkblue") +
  labs(title = "Distribución de Precisión por Participante",
       x = "", y = "Precisión (%)") +
  theme_minimal()





#evolucion de las evaluaciones de los modelos

# Instalar ggplot2 si no lo tienes
# install.packages("ggplot2")

library(ggplot2)

# Datos
evaluaciones <- data.frame(
  Ev = paste0("Ev", 1:17),
  Participantes = c(10,6,6,6,13,14,15,16,17,18,19,20,21,22,23,24,25),
  Precision = c(33,67,50,72,94.49,94.88,94.88,94.95,95.29,95.4,95.64,95.72,95.41,94.77,94.76,95.05,95.07)
)

# Mantener orden de evaluaciones
evaluaciones$Ev <- factor(evaluaciones$Ev, levels = evaluaciones$Ev)

# Crear etiqueta solo con Precision
evaluaciones$Etiqueta <- paste0( evaluaciones$Precision, "%")

ggplot(evaluaciones, aes(x = Ev)) +
  geom_line(aes(y = Precision, group = 1), color = "blue", size = 1) +
  geom_point(aes(y = Precision), color = "blue", size = 3) +
  geom_text(aes(y = Precision, label = Etiqueta), vjust = -0.5, size = 3) +
  labs(title = "Evolución de precisión por evaluación escenario controlado",
       x = "Evaluación",
       y = "Precisión (%)") +
  theme_minimal()



library(ggplot2)

# Datos
evaluaciones <- data.frame(
  Ev = paste0("Ev", 1:17),
  Participantes = c(10,6,6,6,13,14,15,16,17,18,19,20,21,22,23,24,25),
  Precision = c(33,67,50,72,94.49,94.88,94.88,94.95,95.29,95.4,95.64,95.72,95.41,94.77,94.76,95.05,95.07)
)

# Mantener orden de evaluaciones
evaluaciones$Ev <- factor(evaluaciones$Ev, levels = evaluaciones$Ev)

# Gráfico solo con puntos
ggplot(evaluaciones, aes(x = Ev, y = Precision)) +
  geom_point(color = "blue", size = 3) +
  geom_line(aes(group = 1), color = "blue", size = 1) +
  labs(title = "Evolución de precisión por evaluación",
       x = "Evaluación",
       y = "Precisión (%)") +
  theme_minimal()






#no controlados
# Datos
evaluaciones <- data.frame(
  Ev = paste0("Ev", 1:13),
  Participantes = c(13,14,15,16,17,18,19,20,21,22,23,24,25),
  Precision = c(74.12,75.42,74.54,73.57,73.18,72.38,71.82,71.56,71.81,71.27,71.67,72.8,74.38)
)

# Mantener orden de evaluaciones
evaluaciones$Ev <- factor(evaluaciones$Ev, levels = evaluaciones$Ev)

# Crear etiqueta solo con Precision
evaluaciones$Etiqueta <- paste0( evaluaciones$Precision, "%")

ggplot(evaluaciones, aes(x = Ev)) +
  geom_line(aes(y = Precision, group = 1), color = "blue", size = 1) +
  geom_point(aes(y = Precision), color = "blue", size = 3) +
  geom_text(aes(y = Precision, label = Etiqueta), vjust = -0.5, size = 3) +
  labs(title = "Evolución de precisión por evaluación escenario no controlado",
       x = "Evaluación",
       y = "Precisión (%)") +
  theme_minimal()



# si los labels de las precisiones
# Mantener orden de evaluaciones
evaluaciones$Ev <- factor(evaluaciones$Ev, levels = evaluaciones$Ev)

# Gráfico solo con puntos
ggplot(evaluaciones, aes(x = Ev, y = Precision)) +
  geom_point(color = "blue", size = 3) +
  geom_line(aes(group = 1), color = "blue", size = 1) +
  labs(title = "Evolución de precisión por evaluación",
       x = "Evaluación",
       y = "Precisión (%)") +
  theme_minimal()




#ruta_archivo <- "C:\\Users\\Raul\\Desktop\\Proyecto Titulacion\\orientacion1.csv"
#C:\Users\Raul\Desktop\Proyecto Titulacion


ruta_archivo <- "C:\\Users\\Raul\\Desktop\\Proyecto Titulacion\\orientacion1.csv"
ruta_archivo

# Instalar MultBiplotR si no lo tienes
# install.packages("MultBiplotR")

library(MultBiplotR)
library(missMDA)
library(dplyr)

# 1. Leer CSV
ruta_csv <- "C:\\Users\\Raul\\Desktop\\Proyecto Titulacion\\orientacion1.csv" # Cambia esto por la ruta de tu archivo
data <- read.csv(ruta_csv, header = TRUE)

# 2. Extraer IDs de persona
personas <- data$personaid

# 3. Seleccionar solo las columnas numéricas
datos_num <- data %>%
  select(-personaid) %>%
  mutate(across(everything(), as.numeric))

# 4. Estandarización
datos_estandarizados <- scale(datos_num)

# 5. Asignar nombres de fila como personaid
rownames(datos_estandarizados) <- personas

# 6. Crear HJ Biplot
biplot_result <- HJ.Biplot(datos_estandarizados, dimension = 2)

# 7. Graficar
plot(biplot_result,
     Label.rows = TRUE,
     Label.columns = TRUE,
     Color.columns = "green",
     Color.rows = "black",
     Cex.lab = 0.6,
     Show.axes = TRUE)





# ----------------------------------------------------------------------------
# Análisis de agrupaciones mediante PCA
# ----------------------------------------------------------------------------

library(dplyr)
library(FactoMineR)
library(factoextra)

# 1. Leer CSV
ruta_csv <- "C:\\Users\\Raul\\Desktop\\Proyecto Titulacion\\orientacion3.csv"
data <- read.csv(ruta_csv, header = TRUE)

# 2. Extraer IDs de persona
personas <- data$personaid

# 3. Seleccionar solo columnas numéricas
datos_num <- data %>%
  select(-personaid) %>%
  mutate(across(everything(), as.numeric))

# 4. Estandarizar datos
datos_estandarizados <- scale(datos_num)

# 5. Realizar PCA
pca_res <- PCA(datos_estandarizados, graph = FALSE)

# 6. Graficar individuos
fviz_pca_ind(pca_res,
             label = "all",   # Mostrar nombres de todas las personas
             repel = TRUE,    # Evita sobreposición de etiquetas
             col.ind = "blue",
             pointsize = 3)
