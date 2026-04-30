# ============================================================
# CASO: PREDICTING CUSTOMER CHURN AT QWE INC.
# Autores: Cristóbal Herrera e Iván Mayorga
# Curso: Analítica de Negocios - Javeriana 2026
# ============================================================

# 1. PAQUETES
library(readxl)
library(dplyr)
library(ggplot2)
library(stargazer)
library(caret)
library(pscl)

# 2. CARGA Y LIMPIEZA DE DATOS
df <- read_excel("~/Downloads/DATA.xlsx", sheet = "Case Data")

names(df) <- c("ID", "Age", "Churn", "CHI0", "CHI_delta",
               "Cases0", "Cases_delta", "SP0", "SP_delta",
               "Logins", "Blogs", "Views", "DaysSinceLogin")

# 3. TABLA DESCRIPTIVA
df_stats <- as.data.frame(df[, -1])

stargazer(df_stats, type = "text",
          title = "Estadísticas Descriptivas - QWE Inc.", digits = 2)

stargazer(df_stats, type = "html",
          title = "Estadísticas Descriptivas - QWE Inc.", digits = 2,
          out = "tabla_descriptiva.html")

# 4. MODELOS DE REGRESIÓN
formula_modelo <- Churn ~ Age + CHI0 + CHI_delta + Cases0 +
  Cases_delta + SP0 + SP_delta +
  Logins + Blogs + Views + DaysSinceLogin

mpl    <- lm(formula_modelo, data = df)
logit  <- glm(formula_modelo, data = df, family = binomial(link = "logit"))
probit <- glm(formula_modelo, data = df, family = binomial(link = "probit"))

stargazer(mpl, logit, probit, type = "html",
          title = "Modelos de Predicción de Churn - QWE Inc.",
          column.labels = c("MPL", "Logit", "Probit"),
          dep.var.labels = "Churn (1 = Se fue, 0 = Se quedó)",
          digits = 4, star.cutoffs = c(0.10, 0.05, 0.01),
          out = "tabla_regresiones.html")

# 5. R² DE McFADDEN
pR2(logit)

# 6. PREDICCIONES Y MATRIZ DE CONFUSIÓN
df$prob_logit <- predict(logit, type = "response")
df$pred_clase <- ifelse(df$prob_logit > 0.3, 1, 0)

confusionMatrix(as.factor(df$pred_clase),
                as.factor(df$Churn), positive = "1")

# 7. GRÁFICO 1 - Distribución de probabilidades
g1 <- ggplot(df, aes(x = prob_logit, fill = as.factor(Churn))) +
  geom_histogram(bins = 50, alpha = 0.65, position = "identity", color = "white") +
  scale_fill_manual(values = c("#2196F3", "#F44336"),
                    labels = c("No abandonó", "Abandonó")) +
  geom_vline(xintercept = 0.3, linetype = "dashed", color = "black") +
  labs(title = "Distribución de probabilidades predichas de Churn",
       subtitle = "Modelo Logit — QWE Inc.",
       x = "Probabilidad predicha de Churn",
       y = "Número de clientes", fill = "Churn real") +
  theme_minimal()

print(g1)
ggsave("figura1_churn.png", plot = g1, width = 8, height = 5, dpi = 300)

# 8. GRÁFICO 2 - Top 100 clientes en riesgo
top100 <- df %>%
  arrange(desc(prob_logit)) %>%
  slice(1:100) %>%
  mutate(rank = 1:100)

g2 <- ggplot(top100, aes(x = rank, y = prob_logit, color = as.factor(Churn))) +
  geom_point(size = 3) +
  scale_color_manual(values = c("#FF9800", "#F44336"),
                     labels = c("No abandonó", "Sí abandonó")) +
  labs(title = "Top 100 clientes con mayor riesgo de Churn",
       subtitle = "Modelo Logit — QWE Inc.",
       x = "Ranking (1 = mayor riesgo)",
       y = "Probabilidad predicha de Churn",
       color = "Churn real") +
  theme_minimal()

print(g2)
ggsave("figura2_top100.png", plot = g2, width = 8, height = 5, dpi = 300)

# Análisis de Churn - Iván Mayorga

## Contribuciones al proyecto

# Revisión y validación del modelo Logit y Probit
# Interpretación de coeficientes: CHI Score y DaysSinceLogin
# Análisis de la matriz de confusión y limitaciones del modelo
# Revisión de la tabla descriptiva y estadísticas de la muestra

## Hallazgos principales

# El CHI Score es el predictor más relevante de satisfacción
# Los días de inactividad son la señal más temprana de churn
# El modelo Logit presenta mejor ajuste (AIC = 2464.3)

# El modelo sugiere que el churn no ocurre de forma repentina, sino que está precedido por una fase de desconexión gradual del usuario.
