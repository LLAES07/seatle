
# **Descripción del Proyecto**

Este proyecto tiene como objetivo desarrollar un modelo predictivo para estimar el consumo de electricidad de edificios en Seattle (en kWh), utilizando datos públicos de eficiencia energética y características de los edificios. Se exploran diferentes técnicas de machine learning y deep learning, con especial énfasis en la comparación de modelos tradicionales (Random Forest, XGBoost) y redes neuronales (Keras, PyTorch).


# **Datos**

**Fuente:** Data.Seattle.Gov – Building Energy Benchmarking (2018–2023)

**Variables**

| Nombre de la Columna                     | Descripción                                                                                                                                                                | Nombre en la API                    | Tipo de Dato |
|-----------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------|--------------|
| OSEBuildingID                           | Identificador único asignado a cada propiedad cubierta por la Ordenanza de Benchmarking de Seattle para su seguimiento e identificación.                                  | osebuildingid                        | Text         |
| DataYear                                | Año calendario (enero-diciembre) representado por cada registro de datos.                                                                                                  | datayear                             | Text         |
| BuildingName                            | Nombre oficial o común de la propiedad según los registros de la Ciudad de Seattle.                                                                                        | buildingname                         | Text         |
| BuildingType                            | Clasificación general del tipo de edificio según la Ciudad de Seattle.                                                                                                     | buildingtype                         | Text         |
| TaxParcelIdentificationNumber           | Número de identificación de parcela (PIN) del Condado de King.                                                                                                             | taxparcelidentificationnumber        | Text         |
| Address                                 | Dirección física de la propiedad.                                                                                                                                          | address                              | Text         |
| City                                    | Ciudad en la que se encuentra la propiedad.                                                                                                                                | city                                 | Text         |
| State                                   | Estado en el que se encuentra la propiedad.                                                                                                                                | state                                | Text         |
| ZipCode                                 | Código postal de la propiedad.                                                                                                                                             | zipcode                              | Text         |
| Latitude                                | Latitud geográfica de la propiedad.                                                                                                                                        | latitude                             | Number       |
| Longitude                               | Longitud geográfica de la propiedad.                                                                                                                                       | longitude                            | Number       |
| Neighborhood                            | Barrio donde se ubica la propiedad, según el Departamento de Vecindarios de Seattle.                                                                                       | neighborhood                         | Text         |
| CouncilDistrictCode                     | Código del distrito del consejo municipal donde se ubica la propiedad.                                                                                                     | councildistrictcode                 | Number       |
| YearBuilt                               | Año en que se construyó la propiedad.                                                                                                                                      | yearbuilt                            | Text         |
| NumberofFloors                          | Número de pisos habitables por encima del nivel del suelo.                                                                                                                 | numberoffloors                       | Number       |
| NumberofBuildings                       | Número de edificios que forman parte de la propiedad.                                                                                                                      | numberofbuildings                    | Number       |
| PropertyGFATotal                        | Superficie total construida incluyendo estacionamiento, verificada por el gobierno local.                                                                                  | propertygfatotal                     | Number       |
| PropertyGFABuildings                    | Área total construida, excluyendo estacionamientos, verificada por registros públicos.                                                                                     | propertygfabuildings                 | Number       |
| PropertyGFAParking                      | Área total destinada al estacionamiento de todo tipo.                                                                                                                      | propertygfaparking                   | Number       |
| SelfReportGFATotal                      | Superficie total construida y de estacionamiento según el reporte del usuario.                                                                                             | selfreportgfatotal                   | Number       |
| SelfReportGFABuildings                  | Área construida reportada por el usuario, excluye estacionamientos.                                                                                                        | selfreportgfabuildings              | Number       |
| SelfReportParking                       | Área total de estacionamiento según los datos auto-reportados.                                                                                                             | selfreportparking                    | Number       |
| ENERGYSTARScore                         | Puntaje del 1 al 100 calculado por la EPA para evaluar la eficiencia energética total de una propiedad.                                                                   | energystarscore                      | Number       |
| SiteEUIWN(kBtu/sf)                      | Intensidad de uso de energía normalizada por clima, medida en kBtu/pies².                                                                                                  | siteeuiwn_kbtu_sf                    | Number       |
| SiteEUI(kBtu/sf)                        | Intensidad de uso de energía en sitio, medida en kBtu/pies².                                                                                                               | siteeui_kbtu_sf                      | Number       |
| SiteEnergyUse(kBtu)                     | Cantidad total anual de energía consumida por la propiedad de todas las fuentes.                                                                                           | siteenergyuse_kbtu                   | Number       |
| SiteEnergyUseWN(kBtu)                   | Uso total anual de energía ajustado a condiciones climáticas promedio a 30 años.                                                                                           | siteenergyusewn_kbtu                 | Number       |
| SourceEUIWN(kBtu/sf)                    | Intensidad de uso de energía de fuente, normalizada por clima, en kBtu/pies².                                                                                              | sourceeuiwn_kbtu_sf                  | Number       |
| SourceEUI(kBtu/sf)                      | Intensidad de uso de energía de fuente, considerando pérdidas en generación y distribución.                                                                                | sourceeui_kbtu_sf                    | Number       |
| EPAPropertyType                         | Uso principal de la propiedad (ej. oficina, tienda), determinado por la EPA.                                                                                               | epapropertytype                      | Text         |
| LargestPropertyUseType                 | Tipo de uso más grande según superficie construida.                                                                                                                        | largestpropertyusetype              | Text         |
| LargestPropertyUseTypeGFA              | Área construida del tipo de uso más grande.                                                                                                                                | largestpropertyusetypegfa           | Number       |
| SecondLargestPropertyUseType           | Segundo tipo de uso más grande (si aplica).                                                                                                                                | secondlargestpropertyusetype        | Text         |
| SecondLargestPropertyUseTypeGFA        | Área del segundo tipo de uso más grande (si aplica).                                                                                                                       | secondlargestpropertyuse            | Number       |
| ThirdLargestPropertyUseType            | Tercer tipo de uso más grande (si aplica).                                                                                                                                 | thirdlargestpropertyusetype         | Text         |
| ThirdLargestPropertyUseTypeGFA         | Área del tercer tipo de uso más grande (si aplica).                                                                                                                        | thirdlargestpropertyusetypegfa      | Number       |
| Electricity(kWh)                        | Consumo anual de electricidad en kWh (incluye red y sistemas renovables en sitio).                                                                                         | electricity_kwh                     | Number       |
| SteamUse(kBtu)                          | Consumo anual de vapor de distrito, medido en kBtu.                                                                                                                        | steamuse_kbtu                        | Number       |
| NaturalGas(therms)                      | Consumo anual de gas natural suministrado por servicios públicos, en termias.                                                                                              | naturalgas_therms                   | Number       |
| ComplianceStatus                        | Indica si la propiedad cumplió con los requisitos de benchmarking energético.                                                                                              | compliancestatus                     | Text         |
| ComplianceIssue                         | Problemas de cumplimiento conocidos al final del período de gracia.                                                                                                        | complianceissue                      | Text         |
| Electricity(kBtu)                       | Consumo anual de electricidad (red y sistemas renovables) en miles de BTU.                                                                                                 | electricity_kbtu                    | Number       |
| NaturalGas(kBtu)                        | Consumo anual de gas natural en miles de BTU.                                                                                                                               | naturalgas_kbtu                     | Number       |
| TotalGHGEmissions                       | Emisiones totales de gases de efecto invernadero (en toneladas métricas de CO₂e).                                                                                          | totalghgemissions                   | Number       |
| GHGEmissionsIntensity                   | Intensidad de emisiones de GEI por área construida (kgCO₂e/pies²).                                                                                                          | ghgemissionsintensity               | Number       |
| Demolished                              | Indica si la propiedad ha sido demolida al ciclo de reporte de 2023.                                                                                                       | demolished                           | Checkbox     |

# **Preprocesamiento**

**Limpieza**: Eliminación de registros con valores nulos críticos.

**Transformaciones**: Aplicación de log(1 + x) o Box–Cox para normalizar distribuciones muy sesgadas.

**Encoding**: One-Hot para variables categóricas con pocas categorías; Target Encoding para variables de alta cardinalidad.

**Split train/test:** División temporal o aleatoria (80/20), manteniendo la estacionalidad semanal cuando corresponde.



# **Selección de Características**

Análisis de correlación y multicolinealidad (VIF).

Eliminación de variables redundantes (por ejemplo: múltiples EUI con alta correlación).


# **Modelado**

_Modelos Evaluados_

Lineales: Ridge, Lasso, BayesianRidge, sv

Ensamble: RandomForestRegressor, XGBRegressor.


_Validación_

Cross-Validation: RepeatedKFold (5 folds × 3 repeticiones).

Métricas: R², RMSE, MAE en conjunto de validación y test.

Importancia de características basada en Random Forest y XGBoost para refinar el set final.


