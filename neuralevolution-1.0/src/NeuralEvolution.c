/***
 	Universidad de Carabobo
	Facultad Experimental de Ciencias y Tecnologia

	Trabajo Especial de Grado:
		"Algoritmo Genético Paralelo para la Entonación de Parámetros de una
		Red Neuronal Artificial de Retropropagación del Error."

	Autor: 			Orlando Castillo

	Tutor
	Academico:		Joel Rivas

------------------------------------------------------------------------------------------------------
	Archivo:		NeuralEvolution.c

	Descripcion:	Archivo principal del Algoritmo Genético Paralelo para la Entonación de Parámetros
					de una Red Neuronal Artificial de Retropropagación.

	Realizado por:	Orlando Castillo
***/

/* Algoritmo Genetico Paralelo para la entonacion del siguiente conjunto de
 * parametros de una Red Neuronal Artificial entranda por el algoritmo de
 * aprendizaje Retropropagacion del Error:
 *
 * 	- Cantidad de capas ocultas
 * 	- Cantidad de neuronas en cada capa oculta
 * 	- Constante de aprendizaje
 * 	- Razon de momentum
 * 	- Maxima cantidad de iteraciones
 * 	- Funcion de activacion.
 *
 * Para iniciar la ejecucion, el usuario debe especificar 11 parametros obligatorios:
 *
 *	1.- Probabilidad de mutacion
 *	2.- Probabilidad de cruce
 *	3.- Dimension de la poblacion
 *	4.- Cantidad maxima de generaciones
 *  5.- Porcentaje del gap generacional
 *  6.- Cantidad de iteraciones base para el entrenamiento de una RNA
 *  7.- Potencia de diez para el factor multiplicativo
 *  8.- Cantidad de repeticiones de entrenamiento e interrogatorio
 *  9.- Ruta al directorio con los patrones de entrada de las redes
 *  10.- Nombre del archivo con los patrones de entrada (se asume que tiene sufijo .dat)
 *  11.- Tipo de esquema de ejecucion
 *
 * Adicionalmente, existen 3 parametros opcionales que puede especificar en el siguiente orden:
 *
 *	12.- Ruta al directorio en donde se almacenaran los reportes de resultados (Default: './')
 *	13.- Comando para la ejecucion del programa de RNA de retropropagacion
 *       del error que se utilizara para el proceso de optimizacion
 *       (Default: 'RN_BP')
 *  14.- Generar reportes luego del transcurso de una cantidad definida de
 *       generaciones (Default: 0 --> No generar reportes)
 *
*/

	/*****************************************************************/
	/***************   INSTRUCCIONES DE PREPROCESADOR  ***************/
	/*****************************************************************/

/** Librerias **/
#ifdef HAVE_CONFIG_H
	#include <config.h>
#endif

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include <time.h>
#include <ctype.h>
#include "../include/libparallel.h"

/** Constantes **/
#define BITS_CN		5

#if BITS_CN == 7
	#define LONG_CROMOSOMA 	52				//Bits asignados a cada cromosoma
	#define LONG_CANTNEU1 	7				//Bits asignados al parametro cantidad de neuronas en la capa oculta 1 del cromosoma
	#define LONG_CANTNEU2 	7				//Bits asignados al parametro cantidad de neuronas en la capa oculta 2 del cromosoma
	#define MASK_CANTNEU1	(0x7FU)			//Mascara de bits para el segmento de cantidad de neuronas en la capa oculta 1
	#define MASK_CANTNEU2	(0x7FU)			//Mascara de bits para el segmento de cantidad de neuronas en la capa oculta 2
#endif
#if BITS_CN == 6
	#define LONG_CROMOSOMA 	50				//Bits asignados a cada cromosoma
	#define LONG_CANTNEU1 	6				//Bits asignados al parametro cantidad de neuronas en la capa oculta 1 del cromosoma
	#define LONG_CANTNEU2 	6				//Bits asignados al parametro cantidad de neuronas en la capa oculta 2 del cromosoma
	#define MASK_CANTNEU1	(0x3FU)			//Mascara de bits para el segmento de cantidad de neuronas en la capa oculta 1
	#define MASK_CANTNEU2	(0x3FU)			//Mascara de bits para el segmento de cantidad de neuronas en la capa oculta 2
#endif
#if BITS_CN == 5
	#define LONG_CROMOSOMA 	48				//Bits asignados a cada cromosoma
	#define LONG_CANTNEU1 	5				//Bits asignados al parametro cantidad de neuronas en la capa oculta 1 del cromosoma
	#define LONG_CANTNEU2 	5				//Bits asignados al parametro cantidad de neuronas en la capa oculta 2 del cromosoma
	#define MASK_CANTNEU1	(0x1FU)			//Mascara de bits para el segmento de cantidad de neuronas en la capa oculta 1
	#define MASK_CANTNEU2	(0x1FU)			//Mascara de bits para el segmento de cantidad de neuronas en la capa oculta 2
#endif


#define UNO64_T			((uint64_t) 1)	//Constante con el valor 1 en precision de 64 bits sin signo
#define LONG_CANTCAP 	1				//Bits asignados al parametro cantidad de capas ocultas del cromosoma
#define LONG_CONSTAPR 	17				//Bits asignados al parametro contante de aprendizaje del cromosoma
#define LONG_RAZMOM 	10				//Bits asignados al parametro razon de momentum del cromosoma
#define LONG_FACTORITER	8				//Bits asignados al parametro factor de iteraciones del cromosoma
#define LONG_FUNCACT_O	1				//Bits asignados al parametro funcion de activacion de las capas ocultas del cromosoma
#define LONG_FUNCACT_S	1				//Bits asignados al parametro funcion de activacion de la capa de salida del cromosoma

#define MASK_CANTCAP	(0x1U)			//Mascara de bits para el segmento de cantidad de capas ocultas
#define MASK_CONSTAPR	(0x1FFFFU)		//Mascara de bits para el segmento de constante de aprendizaje
#define MASK_RAZMOM		(0x3FFU)		//Mascara de bits para el segmento de razon de momentum
#define MASK_FACTORITER (0xFFU)			//Mascara de bits para el segmento de factor de iteraciones
#define FACTORITER_S 	"128"			//Valor maximo del factor multiplicativo
#define MASK_FUNCACT_O 	(0x1U)			//Mascara de bits para el segmento de funcion de activacion de las capas ocultas
#define MASK_FUNCACT_S 	(0x1U)			//Mascara de bits para el segmento de funcion de activacion de la capa de salida

#define MULTIPLO_AP		1.2				//Valor a utilizar en la renormalizacion

#define DEFAULT_SRUTA	"./"			//Ruta por defecto en donde se almacenaran los resultados
#define DEFAULT_COMANDO	"RN_BP"			//Comando por defecto para la ejecucion de la RNA


/** Macros **/
#define MAX(x, y)				((x) >= (y) ? (x): (y))		//Retorna el mayor de los dos valores proporcionados
#define MIN(x, y)				((x) <= (y) ? (x): (y))		//Retorna el menor de los dos valores proporcionados
#define PRINT_LINE(k, n, file)	for ((k) = 0; (k) < n; (k)++){fputc('-',file);} fputc('\n',file)

	/*****************************************************************/
	/************   DEFINICION DE TIPOS Y ENUMERADOS   ***************/
	/*****************************************************************/

//Tipo de dato logico
typedef enum{false = 0, true = 1} bool;

//Definicion del tipo de dato genotipo. Representa la informacion genotipica de cada inidivido, por lo que depende
//de la definicion previa del tipo cadena_binaria. Adicionalmente, permite la representacion del correspondiente
//valor entero de la cadena para operaciones posteriores.
typedef union{
	uint64_t cromosoma;						//Valor entero que representa la cadena de cromosoma de un individuo
} genotipo_t;

//Definicion del tipo de dato fenotipo. Representa la informacion fenotipica de cada individuo, la cual se obtiene luego de aplicar
//un proceso de decodificacion a la informacion genotipica del mismo.
typedef struct{
	unsigned short int cant_cap;			//Cantidad de capas ocultas
	unsigned short int cant_neu1;			//Cantidad de neuronas en la capa oculta 1
	unsigned short int cant_neu2;			//Cantidad de neuronas en la capa oculta 2
	float const_apr;						//Valor de la constante de aprendizaje
	float raz_mom;							//Valor de la razon de momentum
	unsigned short int func_act_o;			//Funcion de activacion a utilizar por cada neurona de las capas ocultas
	unsigned short int func_act_s;			//Funcion de activacion a utilizar por cada neurona de la capa de salida
	int max_iter;							//Cantidad maxima de iteraciones a utilizar para el entrenamiento de la red
} fenotipo_t;

//Definicion del tipo de dato individuo. Contiene la informacion genotipica y fenotipica correspondiente a cada individuo de
//una poblacion. Adicionalmente, tambien almacena el valor de aptitud para posterior uso
typedef struct{
	genotipo_t genotipo;	//Informacion genotipica
	fenotipo_t fenotipo;	//Informacion fenotipica
	double aptitud;			//Valor de aptitud normalizado al intervalo [0, 1]
	double aptitud_orig;	//Valor de aptitud normalizado al intervalo [0, 1] antes de renormalizar
	int cant_hijos;			//Cantidad de hijos asignados al individuo durante la generacion actual
	genotipo_t padre1;		//Genotipo del primer padre del individuo (en caso de tener)
	genotipo_t padre2;		//Genotipo del segundo padre del individuo (en caso de tener)
	int punto_cruce;		//Punto de cruce seleccionado
	bool cruzo;				//Determina si el individuo proviene de un cruce
	bool muto;				//Determina si el individuo muto
	int pos_pob;			//Posicion del individuo en la poblacion
} individuo_t;

//Definicion del tipo de dato poblacion. Contiene toda la informacion necesaria para procesar una poblacion durante una generacion.
typedef struct{
	individuo_t *individuos;	//Apuntador al arreglo de individuos
	int cant_individuos;		//Cantidad de individuos en la poblacion
	double aptitud_max;			//Aptitud maxima de la poblacion
	double aptitud_prom;		//Aptitud promedio de la poblacion
	double aptitud_min;			//Aptitud minima de la poblacion
	double aptitud_sum;			//Suma de las aptitudes de la poblacion
	int generacion;				//Generacion en la cual se presenta la actual poblacion
} poblacion_t;

//Definicion del tipo de dato datos_t. Contiene informacion util para el proceso de seleccion del algoritmo genetico
typedef struct {
	int indx;		//Indice del individuo en la poblacion
	int hijos;		//Cantidad de hijos asignados
} datos_select_t;

	/****************************************************************/
	/***************   VARIABLES GLOBALES   *************************/
	/****************************************************************/

/** Globales **/

//Parametros del AG
int tam_poblacion;					//Tamanio de una poblacion
int max_gen;						//Cantidad maxima de generaciones del algoritmo genetico
double pc;							//Probabilidad de cruce
double pm;							//Probabilidad de mutacion
double gap;							//Gap generacional
int cant_rep_red;					//Cantidad de sesiones de entrenamiento e interrogatorio para cada instancia de red
int esquema_ejecucion;				//Tipo de esquema de ejecucion seleccionado
int iter_base;						//Cantidad de iteraciones bases para el entrenamiento de cada red
int pot_diez;						//Valor potencia de diez que multiplicara a el factor multiplicativo de cada individuo
char *fruta;						//Ruta al directorio que contiene el archivo con los patrones
char *fnombre;						//Nombre del archivo con los patrones
char *sruta;						//Ruta al directorio en donde se almacenaran los resultados
char *comandoRNA;					//Comando para la ejecucion del programa de RNA de BP
double *vec_max;					//Arreglo con las aptitudes maximas de cada generacion
double *vec_prom;					//Arreglo con las aptitudes promedios de cada generacion
double *vec_min;					//Arreglo con las aptitudes minimas de cada generacion
bool esquema_iniciado = false;		//Determina si el esquema de ejecucion ya fue inicializado
bool parametros_iniciados = false;	//Determina si el los parametros del programa han sido inicializados
time_t time_inicial;				//Tiempo desde donde empieza la ejecucion del programa
int iter_reporte;					//Determina cada cuantas iteraciones generar un mensaje de reporte al usuario
bool debug = true;					//Indica si se desea imprimir mensajes de error por stderr
bool gen_msj = true;				//Indica si se desea imprimir mensajes de error la salida stdout

	/*********************************************************************/
	/***************   PROTOTIPOS DE FUNCIONES   *************************/
	/*********************************************************************/

/**Funciones auxiliares**/
char *parse_arg_ag(int argc, char *argv[]);
char *parametros_entrada(int argc, char *argv[]);
void liberar_parametros();
int cmp_individuo_t(const void *val1, const void *val2);
int cmp_datos_select_t(const void *val1, const void *val2);
double random_uniforme();
int random_intervalo(int a, int b);
bool flip(double p);
void conv_bin_str(uint64_t val, char str[]);
void calcular_escala(double fmin, double fmax, double favg, double *m, double *b);
double escalar_aptitud(double aptitud, double m, double b);
void preparar_esquema_ejecucion();
void liberar_esquema_ejecucion();
void liberar_recursos();
void finalizar_programa(int estado);
void mostrar_error(const char *msj);

/**Funciones del proceso evolutivo**/

//Operaciones a escala de la poblacion
void crear_poblacion(poblacion_t *poblacion, int tam_poblacion);
void liberar_poblacion(poblacion_t *poblacion);
bool buscar_individuo(poblacion_t *poblacion, individuo_t *ind, int ini, int fin);
void insertar_individuo(poblacion_t *poblacion, individuo_t *ind, int pos);
void estadisticas_poblacion(poblacion_t *poblacion);
void iniciar_poblacion(poblacion_t *poblacion);
void muestreo_estocastico_universal(poblacion_t *poblacion, int cant_hijos, datos_select_t *vec_datos);
void seleccion(poblacion_t *poblacion, individuo_t *padre1, individuo_t *padre2, datos_select_t *vec_datos, int *ult_padre);
void generar_nueva_poblacion(poblacion_t *poblacion_act, int cant_reemplazo);
void reemplazar_poblacion(poblacion_t *poblacion_obj, poblacion_t *poblacion_src, int cant_reemplazo);
void aptitudes_poblacion(poblacion_t *poblacion);
void escalar_poblacion(poblacion_t *poblacion);
void algoritmo_genetico(poblacion_t *poblacion);
void reporte_estadisticas(poblacion_t *poblacion, FILE *out);
void reporte_aptitudes(poblacion_t *poblacion, FILE *out);
void solicitar_reporte_mejor_individuo(poblacion_t *poblacion);

//Operaciones a escala de individuos
void generar_genotipo(individuo_t *ind);
void decodificar_genotipo(individuo_t *ind);
void cruce(individuo_t *padre1, individuo_t *padre2, individuo_t *hijo1, individuo_t *hijo2);
void mutacion(individuo_t *individuo);
void solicitar_aptitud(individuo_t *ind);
void calcular_aptitud(individuo_t *ind);

	/*************************************************************************/
	/***************   IMPLEMENTACION DE FUNCIONES   *************************/
	/*************************************************************************/

/**
 * Obtiene los parametros necesarios para el algoritmo genetico
 *
 * Parametros:
 * 		int argc 		- Cantidad de elementos en el arreglo argv
 * 		char *argv[] 	- Arreglo de cadenas de caracteres con la informacion de los parametros proporcionados por terminal
 *
 * Salida
 * 		char *			- Referencia a una cadena caracteres con el mensaje de ocurrencia de un evento, o NULL en caso contrario
 */
char *parse_arg_ag(int argc, char *argv[]){
	//Variables
	int ii;					//Contador
	bool valido = true;		//Validez de la entrada
	char *mensaje = NULL;	//Mensaje de error a retornar si es necesario
	double temp_r;			//Variable real temporal
	int temp_i;				//Variable entera temporal
	int opc = 0;			//Opcion de error
	char *msj[] =
	{
		//
		//Opcion 0
		//
		"              NEURALEVOLUTION\n"
		"     ALGORITMO GENETICO OPTIMIZADOR DE\n"
		"       REDES NEURONALES ARTIFICIALES\n"
		"\n"
		"Para iniciar la ejecucion del programa NeuralEvolution, debe especificar\n"
		"11 parametros obligatorios en el siguiente orden:\n"
		"\n"
		"  1.- Probabilidad de mutacion\n"
		"  2.- Probabilidad de cruce\n"
		"  3.- Dimension de la poblacion\n"
		"  4.- Cantidad maxima de generaciones\n"
		"  5.- Porcentaje del gap generacional\n"
		"  6.- Cantidad de iteraciones base para el entrenamiento de una RNA\n"
		"  7.- Potencia de diez para el factor multiplicativo\n"
		"  8.- Cantidad de repeticiones de entrenamiento e interrogatorio\n"
		"  9.- Ruta al directorio con los patrones de entrada de las redes\n"
		"  10.- Nombre del archivo con los patrones de entrada\n"
		"       (se asume que tiene sufijo .dat)\n"
		"  11.- Tipo de esquema de ejecucion\n"
		"\n"
		"Adicionalmente, existen 3 parametros opcionales que puede especificar en\n"
		"el siguiente orden:\n"
		"\n"
		"  12.- Ruta al directorio en donde se almacenaran los reportes de\n"
		"       resultados (Default: './')\n"
		"  13.- Comando para la ejecucion del programa de RNA de retropropagacion\n"
		"       del error que se utilizara para el proceso de optimizacion\n"
		"       (Default: 'RN_BP')\n"
		"  14.- Generar reportes luego del transcurso de una cantidad definida de\n"
		"       generaciones (Default: 0 --> No generar reportes)\n"
		"\n"
		"Para mayor informacion, ejecute el programa con la opcion --help (-h)\n",
		//
		//Opcion 1
		//
		"La probabilidad de mutacion debe ser un valor en el intervalo real [0,1]\n",
		//
		//Opcion 2
		//
		"La probabilidad de cruce debe ser un valor en el intervalo real [0,1]\n",
		//
		//Opcion 3
		//
		"La dimension de la poblacion debe ser un valor entero positivo\n",
		//
		//Opcion 4
		//
		"La dimension de la poblacion debe ser un valor par\n",
		//
		//Opcion 5
		//
		"La cantidad maxima de generaciones debe ser un valor entero positivo\n",
		//
		//Opcion 6
		//
		"El gap generacional debe ser un valor en el intervalo real [0,1]\n",
		//
		//Opcion 7
		//
		"La cantidad de iteraciones base para el entrenamiento de la RNA debe ser un\n"
		"valor entero positivo\n",
		//
		//Opcion 8
		//
		"La potencia de diez para el factor multiplicativo debe ser un valor entero\n"
		"positivo potencia de 10\n",
		//
		//Opcion 9
		//
		"El valor de la potencia de diez es muy grande ya que el valor maximo de\n"
		"factor multiplicativo es "FACTORITER_S"\n" ,
		//
		//Opcion 10
		//
		"La cantidad de repeticiones de entrenamiento e interrogatorio debe ser\n"
		"un valor entero positivo\n",
		//
		//Opcion 11
		//
		"El valor asociado al esquema de ejecucion debe ser un entero en\n"
		"el intervalo [1, 3]\n",
		//
		//Opcion 12
		//
		"Si desea generar reportes, entonces debe especificar un valor entero no\n"
		"negativo menor o igual que la cantidad de generaciones especificada\n"
		"anteriormente, o el valor 0 si no desea reportes.",
	};		//Mensajes de error posibles

	//La cantidad de argumentos no coincide con la requerida
	if (argc < 11 || argc > 14){
		valido = false;
		opc = 0;
	}

	//Inicializacion obligatoria de algunas variables
	fruta = NULL;
	fnombre = NULL;
	sruta = NULL;
	comandoRNA  = NULL;
	iter_reporte = 0;

	//Reocorrido del vector de parametros obligatorio, validando el formato de los mismos
	for (ii = 0; ii < 11 && valido; ii++){
		switch(ii){
			//Probabilidad de mutacion
			case 0:{
				temp_r = strtod(argv[ii], NULL);
				if (temp_r >= 0.0 && temp_r <= 1.0)
					pm = temp_r;
				else{
					valido = false;
					opc = 1;
				}
				break;
			}
			//Probabilidad de cruce
			case 1:{
				temp_r = strtod(argv[ii], NULL);
				if (temp_r >= 0.0 && temp_r <= 1.0)
					pc = temp_r;
				else{
					valido = false;
					opc = 2;
				}
				break;
			}
			//Tamanio de la poblacion
			case 2:{
				temp_i = atoi(argv[ii]);
				if (temp_i > 0){
					if (temp_i % 2 == 0)
						tam_poblacion = temp_i;
					else{
						valido = false;
						opc = 4;
					}
				}
				else{
					valido = false;
					opc = 3;
				}
				break;
			}
			//Maxima cantidad de generaciones
			case 3:{
				temp_i = atoi(argv[ii]);
				if (temp_i > 0)
					max_gen = temp_i;
				else{
					valido = false;
					opc = 5;
				}
				break;
			}
			//Porcentaje de gap generacional
			case 4:{
				temp_r = strtod(argv[ii], NULL);
				if (temp_r >= 0.0 && temp_r <= 1.0)
					gap = temp_r;
				else{
					valido = false;
					opc = 6;
				}
				break;
			}
			//Cantidad de iteraciones base
			case 5:{
				temp_i = atoi(argv[ii]);
				if (temp_i > 0)
					iter_base = temp_i;
				else{
					valido = false;
					opc = 7;
				}
				break;
			}
			//Potencia de diez
			case 6:{
				int es_pot = 1, pot10;

				//Verificacion de que el valor sea potencia de diez y positivo
				temp_i = atoi(argv[ii]);
				if (temp_i <= 0)
					es_pot = 0;
				else{
					pot10 = 1;
					while (pot10 <= temp_i)
						pot10 *= 10;
					pot10 /= 10;
					if (pot10 != temp_i)
						es_pot = 0;
				}

				if (es_pot){
					pot_diez = temp_i;
					if ((pot_diez * (MASK_FACTORITER >> 1)) > iter_base){
						valido = false;
						opc = 9;
					}
				}
				else{
					valido = false;
					opc = 8;
				}
				break;
			}
			//Cantidad de repeticiones de entrenamiento e interrogatorio
			case 7:{
				temp_i = atoi(argv[ii]);
				if (temp_i > 0)
					cant_rep_red = temp_i;
				else{
					valido = false;
					opc = 10;
				}
				break;
			}
			//Ruta al directorio con el archivo de patrones
			case 8:{
				fruta = (char *) calloc(strlen(argv[ii]) + 10, sizeof(char));
				sprintf(fruta,"%s",argv[ii]);
				break;
			}
			//Nombre del archivo con los patrones
			case 9:{
				fnombre = (char *) calloc(strlen(argv[ii]) + 5, sizeof(char));
				strcpy(fnombre, argv[ii]);
				break;
			}
			//Tipo de esquema paralelos
			case 10:{
				temp_i = atoi(argv[ii]);
				if (temp_i >= 1 && temp_i <= CANT_ESQUEMAS){
					switch(temp_i){
						//Esquema secuencial
						case 1:
							esquema_ejecucion = SECUENCIAL;
							break;
						//Esquema con fork
						case 2:
							esquema_ejecucion = FORK;
							break;
						default:
							esquema_ejecucion = MPI;
							break;
					}
				}
				else{
					valido = false;
					opc = 11;
				}
				break;
			}
		}
	}

	//Procesamiento de los parametros opcionales, verificando el formato de los mismos
	for (ii = 11; ii < argc && valido; ii++){
		switch(ii){
			//Ruta al directorio donde se almacenaran los reportes
			case 11:{
				sruta = (char *) calloc(strlen(argv[ii]) + 5, sizeof(char));
				strcpy(sruta, argv[ii]);
				break;
			}
			//Comando para la ejecucio de la RNA
			case 12:{
				comandoRNA = (char *) calloc(strlen(argv[ii]) + 5, sizeof(char));
				strcpy(comandoRNA, argv[ii]);
				break;
			}
			//Generacion de reportes cada cierta cantidad de generaciones
			case 13:{
				temp_i = atoi(argv[ii]);
				if (temp_i >= 0 && temp_i <= max_gen)
					iter_reporte = temp_i;
				else{
					valido = false;
					opc = 12;
				}
				break;
			}
		}
	}

	//Se presento un error en los datos porporcionados
	if (!valido){
		mensaje = (char *) malloc(strlen(msj[opc]) + 21);
		strcpy(mensaje, msj[opc]);

		//Liberacion de variables
		if (fruta != NULL)
			free(fruta);
		if (fnombre != NULL)
			free(fnombre);
		if (sruta != NULL)
			free(sruta);
		if (comandoRNA != NULL)
			free(comandoRNA);
	}
	else{
		//Inicializacion de la memoria para los vectores con las aptitudes por generacion
		vec_max = (double *) calloc(max_gen + 5, sizeof(double));
		vec_prom = (double *) calloc(max_gen + 5, sizeof(double));
		vec_min = (double *) calloc(max_gen + 5, sizeof(double));

		//Si no se especificaron los parametros opcionales, se les asignan los valores por defecto
		if (comandoRNA == NULL){
			comandoRNA = (char *) calloc(strlen(DEFAULT_COMANDO) + 5, sizeof(char));
			strcpy(comandoRNA, DEFAULT_COMANDO);
		}
		if (sruta == NULL){
			sruta = (char *) calloc(strlen(DEFAULT_SRUTA) + 5, sizeof(char));
			strcpy(sruta, DEFAULT_SRUTA);
		}

		//Se registra la inicializacion
		parametros_iniciados = true;
	}

	//Retorno del apuntador a la cadena de error si es que existe
	return mensaje;
}

/**
 * Asignacion de valores a los parametros especificados por la cadena de caracteres argv
 *
 * Parametros:
 * 		int argc 		- Cantidad de elementos en el arreglo argv
 * 		char *argv[] 	- Arreglo de cadenas de caracteres con la informacion de los parametros proporcionados por terminal
 *
 * Salida
 * 		char *			- Referencia a una cadena caracteres con el mensaje de ocurrencia de un evento, o NULL en caso contrario
 */
char *parametros_entrada(int argc, char *argv[]){
	//Variables
	int ii;					//Contador
	bool valido = true;		//Validez de la entrada
	char *msj = NULL;		//Mensaje de error a retornar si es necesario
	int opc = 0;			//Opcion de error
	char **vec_arg;			//Vector con los argumentos del programa
	int cant_arg;			//Cantidad de argumentos proporcionados
	char val_opc;			//Valor de una opcion especificada como argumento
	char *err_pos[] =
	{
		//
		//Opcion 0
		//
		"              NEURALEVOLUTION\n"
		"     ALGORITMO GENETICO OPTIMIZADOR DE\n"
		"       REDES NEURONALES ARTIFICIALES\n"
		"\n"
		"Para iniciar la ejecucion del programa NeuralEvolution, debe especificar\n"
		"11 parametros obligatorios en el siguiente orden:\n"
		"\n"
		"  1.- Probabilidad de mutacion\n"
		"  2.- Probabilidad de cruce\n"
		"  3.- Dimension de la poblacion\n"
		"  4.- Cantidad maxima de generaciones\n"
		"  5.- Porcentaje del gap generacional\n"
		"  6.- Cantidad de iteraciones base para el entrenamiento de una RNA\n"
		"  7.- Potencia de diez para el factor multiplicativo\n"
		"  8.- Cantidad de repeticiones de entrenamiento e interrogatorio\n"
		"  9.- Ruta al directorio con los patrones de entrada de las redes\n"
		"  10.- Nombre del archivo con los patrones de entrada\n"
		"       (se asume que tiene sufijo .dat)\n"
		"  11.- Tipo de esquema de ejecucion\n"
		"\n"
		"Adicionalmente, existen 3 parametros opcionales que puede especificar en\n"
		"el siguiente orden:\n"
		"\n"
		"  12.- Ruta al directorio en donde se almacenaran los reportes de\n"
		"       resultados (Default: './')\n"
		"  13.- Comando para la ejecucion del programa de RNA de retropropagacion\n"
		"       del error que se utilizara para el proceso de optimizacion\n"
		"       (Default: 'RN_BP')\n"
		"  14.- Generar reportes luego del transcurso de una cantidad definida de\n"
		"       generaciones (Default: 0 --> No generar reportes)\n"
		"\n"
		"Para mayor informacion, ejecute el programa con la opcion --help (-h)\n",
		//
		//Opcion 1
		//
		"              NEURALEVOLUTION\n"
		"     ALGORITMO GENETICO OPTIMIZADOR DE\n"
		"       REDES NEURONALES ARTIFICIALES\n"
		"\n"
		"Modo de uso: NeuralEvolution [opciones] [argumentos]\n"
		"\n"
		"Algoritmo Genetico optimizador de Redes Neuronales Artificiales entrenadas\n"
		"por el algoritmo de aprendizaje Retropropagacion del Error.\n"
		"\n"
		"  * OPCIONES DISPONIBLES:\n"
		"\n"
		"     --help (-h)      Muestra este mensaje de ayuda.\n"
		"\n"
		"     --version        Version del programa y datos de autores.\n"
		"\n"
		"     --debug=[y/n]    Indica si se desean mostrar los mensajes de\n"
		"                      errores (y = 'si', n = 'no') Default: 'y'\n"
		"\n"
		"     --mensajes=[y/n] Indica si se desea que el programa imprima\n"
		"                      mensajes por pantalla con reportes de estado.\n"
		"                      (y = 'si', n = 'no') Default: 'y'\n"
		"\n"
		"  * ARGUMENTOS VALIDOS\n"
		"\n"
		"Para iniciar la ejecucion del programa NeuralEvolution, debe especificar\n"
		"11 parametros obligatorios en el siguiente orden:\n"
		"\n"
		"  1.- Probabilidad de mutacion\n"
		"\n"
		"           Valor real dentro del intervalo [0, 1] que indica la\n"
		"           probabilidad de que un individuo sea sometido al\n"
		"           operador de mutacion.\n"
		"\n"
		"  2.- Probabilidad de cruce\n"
		"\n"
		"           Valor real dentro del intervalo [0, 1] que indica la\n"
        "           probabilidad de que un par de individuos sean\n"
		"           sometidos al operador de cruce.\n"
		"\n"
		"  3.- Dimension de la poblacion\n"
		"\n"
		"           Valor entero positivo par que especifica la cantidad de\n"
		"           individuos dentro de cada poblacion del algoritmo genetico\n"
		"\n"
		"  4.- Cantidad maxima de generaciones\n"
		"\n"
		"           Valor entero positivo que especifica la cantidad de generaciones\n"
		"           que deben transcurrir para culminar la ejecucion\n"
		"\n"
		"  5.- Porcentaje del gap generacional\n"
		"\n"
		"           Valor real dentro del intervalo [0, 1] que indica el\n"
		"           porcentaje de individuos que seran reemplazados en cada\n"
		"           generacion por la descendencia\n"
		"\n"
		"  6.- Cantidad de iteraciones base para el entrenamiento de una RNA\n"
		"\n"
		"           Valor entero positivo que especifica la cantidad de iteraciones\n"
		"           base que sera utilizada para calcular luego la cantidad maxima\n"
		"           de iteraciones en el entrenamiento de cada red\n"
		"\n"
		"  7.- Potencia de diez para el factor multiplicativo\n"
		"\n"
		"           Valor entero positivo potencia de diez que sera utilizado\n"
		"           luego para calcular la cantidad maxima\n"
		"           de iteraciones en el entrenamiento de cada red\n"
		"\n"
		"  8.- Cantidad de repeticiones de entrenamiento e interrogatorio\n"
		"\n"
		"           Valor entero positivo que indica la cantidad de sesiones\n"
		"           de entrenamiento e interrogatorio a la que seran sometida\n"
		"           cada una de las redes.\n"
		"\n"
		"  9.- Ruta al directorio con los patrones de entrada de las redes\n"
		"\n"
		"           Ruta relativa o absoluta al directorio que contiene\n"
		"           el archivo con los patrones de entrada a usar en\n"
		"           el aprendizaje de cada red. La ruta debe obligatoriamente\n"
		"           finalizar con el caracter delimitador de ruta\n"
		"           respectivo del sistema (Linux: '/')\n"
		"\n"
		"  10.- Nombre del archivo con los patrones de entrada\n"
		"\n"
		"           Nombre del archivo sin sufijo de tipo con los\n"
		"           patrones de entrada a usar en el interrogatorio.\n"
		"           Se asume que el archivo se encuentra en el\n"
		"           directorio especificado en el argumento\n"
		"           anterior, es de tipo '.dat' y que cumple con\n"
		"           el formato requerido, el cual se especifica\n"
		"           en el manual proporcionado\n"
		"\n"
		"  11.- Tipo de esquema de ejecucion\n"
		"\n"
		"           Indica el esquema de ejecucion bajo el cual se\n"
		"           ejecutara el programa. El interes es permitir que\n"
		"           el programa pueda ejecutarse bajo diversos paradigmas,\n"
		"           en especial el paralelo. Las opciones disponibles\n"
		"           son las siguientes:\n"
		"\n"
		"           1       -->   Esquema de ejecucion completamente\n"
		"                         secuencial\n"
		"\n"
		"           2       -->   Esquema de ejecucion con fork utilizando\n"
		"                         memoria compartida. \n"
		"\n"
		"           3       -->   Esquema de ejecucion con mpi 2.\n"
		"\n"
		"Adicionalmente, existen 3 parametros opcionales que puede especificar en\n"
		"el siguiente orden:\n"
		"\n"
		"  12.- Ruta al directorio en donde se almacenaran los reportes de\n"
		"       resultados\n"
		"\n"
		"           Ruta relativa o absoluta al directorio en donde se guardaran\n"
		"           los reportes de estadisticas generados. La ruta debe\n"
		"           obligatoriamente finalizar con el caracter delimitador\n"
		"           de ruta respectivo del sistema (Linux: '/')\n"
		"           Default: Directorio actual (Linux: './')\n"
		"\n"
		"  13.- Comando para la ejecucion del programa de RNA de retropropagacion\n"
		"       del error que se utilizara para el proceso de optimizacion\n"
		"\n"
		"           Programa de RNA a utilizar por el algoritmo genetico\n"
		"           para evaluar el rendimiento de una arquitectura.\n"
		"           Default: RN_BP\n"
		"\n"
		"  14.- Generar reportes luego del transcurso de una cantidad definida de\n"
		"       generaciones\n"
		"\n"
		"           Valor entero no negativo que le indica al programa cada\n"
		"           cuantas generaciones se desea generar un reporte\n"
		"           de datos. En caso de especificar 0,o si se proporciona\n"
		"           la opcion 'mensajes=n' al programa, no se imprimira ningun\n"
		"           reporte. Default: 0\n"
		"\n"
		"Para mayor informacion, revisar el manual de usuario distribuido con\n"
		"el software\n",
		//
		//Opcion 2
		//
		"NeuralEvolution 1.0\n"
		"Algoritmo Genetico optimizador de Redes Neuronales entrenadas\n"
		"por el algoritmo de aprendizaje Retropropagacion del Error.\n"
		"\n"
		"Desarrollado por: Orlando Castillo\n"
		"Universidad de Carabobo, Venezuela - 2012\n",
		//
		//Opcion 3
		//
		"Opcion proporcionada invalida. Ejecute el programa con la opcion\n"
		"--help (-h) para obtener mayor informacion\n",
		//
		//Opcion 4
		//
		"Valor invalido para la opcion --debug. Esta opcion solo tiene dos posibles\n"
		"valores: 'y' o 'n'. Ejecute el programa con la opcion --help (-h) para\n"
		"mayor informacion\n",
		//
		//Opcion 5
		//
		"Valor invalido para la opcion --mensajes. Esta opcion solo tiene dos posibles\n"
		"valores: 'y' o 'n'. Ejecute el programa con la opcion --help (-h) para\n"
		"mayor informacion\n"
	};

	//No se especificaron parametros
	if (argc == 1){
		valido = false;
		opc = 0;
	}
	//Al menos existe un parametro
	else{
		//Creacion de espacio de memoria
		vec_arg = (char **) calloc(argc, sizeof(char *));
		cant_arg = 0;

		//Busqueda de las opciones en el vector de argumentos, verificando que sean opciones validas
		//y creando un vector con los argumentos sin las opciones
		for (ii = 1; ii < argc && valido; ii++){
			//Es una posible opcion valida
			if ((argv[ii][0] == '-') && (strlen(argv[ii]) > 1) && (!isdigit(argv[ii][1]))){
				if ((strcmp(argv[ii], "-h") == 0) || (strcmp(argv[ii], "--help") == 0)){
					valido = false;
					opc = 1;
				}
				else if (strcmp(argv[ii], "--version") == 0){
					valido = false;
					opc = 2;
				}
				else if (strncmp(argv[ii], "--debug=", 8) == 0){
					if (strlen(argv[ii]) != 9){
						valido = false;
						opc = 4;
					}
					else{
						val_opc = argv[ii][8];
						if (val_opc == 'y')
							debug = true;
						else if (val_opc == 'n')
							debug = false;
						else{
							valido = false;
							opc = 4;
						}
					}
				}
				else if (strncmp(argv[ii], "--mensajes=", 11) == 0){
					if (strlen(argv[ii]) != 12){
						valido = false;
						opc = 5;
					}
					else{
						val_opc = argv[ii][11];
						if (val_opc == 'y')
							gen_msj = true;
						else if (val_opc == 'n')
							gen_msj = false;
						else{
							valido = false;
							opc = 5;
						}
					}
				}
				else{
					valido = false;
					opc = 3;
				}
			}
			//Posible argumento valido
			else{
				vec_arg[cant_arg] = (char *) calloc(strlen(argv[ii]) + 5, sizeof(char));
				strcpy(vec_arg[cant_arg++], argv[ii]);
			}
		}

		//No hay argumentos
		if (valido && cant_arg == 0){
			valido = false;
			opc = 0;
		}

		//Sigue siendo valido
		if (valido)
			msj = parse_arg_ag(cant_arg, vec_arg);

		//Liberacion
		for (ii = 0; ii < cant_arg; ii++)
			free(vec_arg[ii]);
		free(vec_arg);
	}

	//Se presento un error en los datos proporcionados
	if (!valido){
		msj = (char *) malloc(strlen(err_pos[opc]) + 20);
		strcpy(msj, err_pos[opc]);
	}

	//Retorno del apuntador a la cadena de error si es que existe
	return msj;
}

/**
 * Liberar el espacio ocupado por los parametros de entrada
 */
void liberar_parametros(){
	free(fruta);
	free(fnombre);
	free(sruta);
	free(comandoRNA);
	free(vec_max);
	free(vec_prom);
	free(vec_min);

	//Se registra la liberacion
	parametros_iniciados = false;
}


/**
 * Funcion de comparacion para variables de tipo individuo_t. El principal proposito
 * es utilizar esta funcion como parametros a las funciones de libreria qsort
 *
 * Parametros:
 * 		const void *val1 		- Referencia generica al primer valor
 * 		const void *val1 		- Referencia generica al segundo valor
 */
int cmp_individuo_t(const void *val1, const void *val2){
	//Variables
	individuo_t *ind1, *ind2;	//Individuos a comparar

	//Casteo a los tipos correctos
	ind1 = (individuo_t *) val1;
	ind2 = (individuo_t *) val2;

	//Proceso de comparacion

	//Individuo 1 menor al 2
	if (ind1->aptitud < ind2->aptitud)
		return -1;

	//Individuo 1 mayor al 2
	if (ind1->aptitud > ind2->aptitud)
		return 1;

	//Individuos iguales
	return 0;
}

/**
 * Funcion de comparacion para variables de tipo datos_select_t. El principal proposito
 * es utilizar esta funcion como parametros a las funciones de libreria qsort
 *
 * Parametros:
 * 		const void *val1 		- Referencia generica al primer valor
 * 		const void *val1 		- Referencia generica al segundo valor
 */
int cmp_datos_select_t(const void *val1, const void *val2){
	//Variables
	datos_select_t *dat1, *dat2;	//Individuos a comparar

	//Casteo a los tipos correctos
	dat1 = (datos_select_t *) val1;
	dat2 = (datos_select_t *) val2;

	//Proceso de comparacion

	//Valores iguales
	if (dat1->hijos == dat2->hijos)
		return 0;

	//dat1 es menor que dat2, por lo que deberia ir luego de dat2
	if (dat1->hijos < dat2->hijos)
		return 1;

	//dat1 es mayor que dat2, por lo que deberia ir antes de dat2
	return -1;
}

/**
 * Generacion de un valor aleatorio uniforme en el intervalo real [0, 1]
 *
 * Salida
 * 		double			- Valor aleatorio generado
 */
double random_uniforme(){
	return (rand() / (RAND_MAX * 1.0));
}

/**
 * Generacion de un valor entero uniforme en el intervalo entero [a, b]. Se asume que b >= a
 *
 * Parametros:
 * 		int a		- Extremo inicial del intervalo
 * 		int b		- Extremo final del intervalo
 */
int random_intervalo(int a, int b){
	int offset = (b - a + 1) * (rand() / (RAND_MAX + 1.0));
	return a + offset;
}

/**
 * Simulacion del lanzamiento de una moneda con probabilidad p de obtener cara
 *
 * Parametros:
 * 		double p		- Probabilidad de la ocurrencia de conseguir cara
 *
 * Salida
 * 		bool			- Retorna true si se obtuvo cara, false de lo contrario
 */
bool flip(double p){
	if (random_uniforme() <= p)
		return true;
	return false;
}

/**
 * Conversion de a representacion binaria de un entero proporcionado a una cadena
 *
 * Parametros:
 * 		uint64_t val		- Valor a convertir
 * 		char str[]			- Cadena con el resultado de la conversion
 */
void conv_bin_str(uint64_t val, char str[]){
	//Variables
	int ii, jj; 		//Contadores

	for (ii = 0, jj = (LONG_CROMOSOMA - 1); jj >= 0; ii++, jj--){
		if (val & (UNO64_T << ii))
			str[jj] = '1';
		else
			str[jj] = '0';
	}
	str[LONG_CROMOSOMA] = '\0';
}

/**
 * Calcula los coeficiente de escala para la renormalizacion lineal
 *
 * Parametros:
 * 		double fmin			- Minimo valor de aptitud de la poblacion
 * 		double fmax			- Maximo valor de aptitud de la poblacion
 * 		double favg			- Aptitud promedio de la poblacion
 * 		double *m			- Referencia a la variable que almacenara la pendiente de la recta de renormalizacion
 * 		double *b			- Referencia a la variable que almacenara la interseccion en y de la recta de renormalizacion
 *
 */
void calcular_escala(double fmin, double fmax, double favg, double *m, double *b){
	//Variables
	double delta;

	//Verificacion de que ninguna aptitud pueda ser negativa
	if (fmin > ((MULTIPLO_AP * favg - fmax) / (MULTIPLO_AP - 1.0))){
		delta = fmax - favg;
		*m = ((MULTIPLO_AP - 1.0) * favg) / delta;
		*b = favg * ((fmax - MULTIPLO_AP * favg) / delta);
	}
	//Se escala tanto como se pueda en caso contrario
	else{
		if (favg == fmin){
			*m = 1.0;
			*b = 0;
		}
		else{
			delta = favg - fmin;
			*m = favg / delta;
			*b = (-fmin * favg) / delta;
		}
	}
}

/**
 * Escala el valor de aptitud proporcionado utilizando una recta de renormalizacion
 *
 * Parametros:
 * 		double aptitud		- Valor de aptitud a escalar
 * 		double m			- Pendiente de la recta de renormalizacion
 * 		double b			- Interseccion en y de la recta de renormalizacion
 *
 * Retorno:
 * 		double				- Valor de aptitud normalizado
 *
 */
double escalar_aptitud(double aptitud, double m, double b){
	return (m * aptitud + b);
}

/**
 * Preparacion del esquema de ejecucion para el AG maestro
 *
 */
void preparar_esquema_ejecucion(){
	switch(esquema_ejecucion){
		//Esquema con fork
		case FORK:{
			//Preparacion del esquema paralelo con fork
			if (!preparar_fork_maestro(tam_poblacion)){
				mostrar_error("No pudieron prepararse los recursos necesarios para el esquema de ejecucion con fork\n");
				finalizar_programa(-1);
			}
			break;
		}
		//Esquema con mpi
		case MPI:{
			//Preparacion del esquema paralelo con fork
			if (!preparar_mpi_maestro(tam_poblacion)){
				mostrar_error("No pudieron prepararse los recursos necesarios para el esquema de ejecucion con mpi\n"
						"Asegurese de haber configurado la aplicacion con soporte para mpi (./configure --with-mpi=yes)\n");
				finalizar_programa(-1);
			}
			break;
		}
		//Esquema secuencial (NO RECOMENDADO)
		case SECUENCIAL: default:{
			break;		//No hace falta inicializar recursos
		}
	}
	//Se registra la inicializacion
	esquema_iniciado = true;
}

/**
 * Liberacion de los recursos del esquema de ejecucion preparado para el AG maestro
 *
 */
void liberar_esquema_ejecucion(){
	//Se registra la liberacion
	esquema_iniciado = false;
	switch(esquema_ejecucion){
		//Esquema con fork
		case FORK:{
			//Liberacion del esquema paralelo con fork
			if (!liberar_fork()){
				mostrar_error("No pudieron liberarse los recursos necesarios para el esquema de ejecucion con fork\n");
				finalizar_programa(-1);
			}
			break;
		}
		//Esquema con mpi
		case MPI:{
			//Liberacion del esquema paralelo con fork
			if (!liberar_mpi()){
				mostrar_error("No pudieron liberarse los recursos necesarios para el esquema de ejecucion con mpi\n");
				finalizar_programa(-1);
			}
			break;
		}
		//Esquema secuencial (NO RECOMENDADO)
		case SECUENCIAL: default:{
			break; 		//No hace falta liberar recursos
		}
	}
}

/**
 * Liberacion de recursos dinamicos
 *
 */
void liberar_recursos(){
	//Liberacion de la memoria asignada a los patrones
	if (parametros_iniciados)
		liberar_parametros();

	//Liberacion del esquema de ejecucion
	if (esquema_iniciado)
		liberar_esquema_ejecucion();
}

/**
 * Permite terminar la ejecucion del programa, considerando la limpieza de los recursos dinamicos utilizados
 *
 */
void finalizar_programa(int estado){
	//Liberacion de recursos dinamicos
	liberar_recursos();

	//Finaliza la ejecucion
	exit(estado);
}

//Impresion de un mensaje por la salida stderr solo si esta activada la opcion debug
void mostrar_error(const char *msj){
	if (debug)
		fprintf(stderr, "%s", msj);
}

/**
 * Reserva de la cantidad de memoria necesaria para almacenar a los individuos de una poblacion
 *
 * Parametros:
 * 		poblacion_t *poblacion		- Poblacion a la cual se le reservara la memoria
 * 		int tam_poblacion			- Cantidad de individuos que componen a una poblacion
 */
void crear_poblacion(poblacion_t *poblacion, int tam_poblacion){
	//Variables
	int ii;		//Contador

	//Inicializacion
	poblacion->generacion = 0;
	poblacion->cant_individuos = tam_poblacion;
	poblacion->individuos = (individuo_t *) malloc(tam_poblacion * sizeof(individuo_t));
	for (ii = 0; ii < tam_poblacion; ii++){
		poblacion->individuos[ii].aptitud = 0.0;
		poblacion->individuos[ii].aptitud_orig = 0.0;
		poblacion->individuos[ii].genotipo.cromosoma = 0;
		poblacion->individuos[ii].pos_pob = ii;
	}

}

/**
 * Liberacion de la cantidad de memoria reservada para una poblacion
 *
 * Parametros:
 * 		poblacion_t *poblacion		- Poblacion que se desea liberar
 */
void liberar_poblacion(poblacion_t *poblacion){
	free(poblacion->individuos);
}

/**
 * Busqueda del individuo en la poblacion dentro del intervalo [ini, fin]. Si lo consigue,
 * retorna true, en caso contrario retorna false.
 *
 * Parametros:
 * 		poblacion_t *poblacion		- Poblacion donde se realizara la busqueda
 * 		individuo_t *ind			- Individuo a buscar
 * 		int ini						- Extremo inicial (indexado en cero) del intervalo de busqueda
 * 		int fin						- Extremo final (indexado en cero) del intervalo de busqueda
 *
 * Salida
 * 		bool 						- Resultado de la busqueda
 */
bool buscar_individuo(poblacion_t *poblacion, individuo_t *ind, int ini, int fin){
	//Variables
	int ii;				//Contador
	bool encontrado;	//Booleano que registra el resultado de la busqueda

	//Busqueda de un individuo con el mismo cromosoma en el intervalo proporcionado mientras no se consiga
	encontrado = false;
	for (ii = ini; ii <= fin && !encontrado; ii++){
		if (poblacion->individuos[ii].genotipo.cromosoma == ind->genotipo.cromosoma)
			encontrado = true;
	}
	return encontrado;
}

/**
 * Inserta un individuo dentro de la poblacion en la posicion especificada
 *
 * Parametros:
 * 		poblacion_t *poblacion		- Poblacion donde se insertara
 * 		individuo_t *ind			- Individuo a insertar
 * 		int pos						- Posicion (indexada en cero) donde se insertara el individuo
 */
void insertar_individuo(poblacion_t *poblacion, individuo_t *ind, int pos){
	//Insercion del individuo
	poblacion->individuos[pos] = *ind;
}

/**
 * Generacion de diversas estadisticas de la poblacion
 *
 * Parametros:
 * 		poblacion_t *poblacion		- Poblacion a la cual se calcularan estadisticas
 */
void estadisticas_poblacion(poblacion_t *poblacion){
	//Variables
	int ii;			//Contador

	//Inicializacion de algunos campos
	poblacion->aptitud_max = -1.0;
	poblacion->aptitud_min = 2.0;
	poblacion->aptitud_sum = 0.0;

	//Calculo de estadisticas
	for (ii = 0; ii < poblacion->cant_individuos; ii++){
		poblacion->aptitud_max = MAX(poblacion->aptitud_max, poblacion->individuos[ii].aptitud_orig);
		poblacion->aptitud_min = MIN(poblacion->aptitud_min, poblacion->individuos[ii].aptitud_orig);
		poblacion->aptitud_sum += poblacion->individuos[ii].aptitud_orig;
	}
	poblacion->aptitud_prom = poblacion->aptitud_sum / poblacion->cant_individuos;
}

/**
 * Inicializacion aleatoria de una poblacion de individuos no repetidos
 *
 * Parametros:
 * 		poblacion_t *poblacion		- Poblacion a inicializar, la cual se asume ya creada
 */
void iniciar_poblacion(poblacion_t *poblacion){
	//Variables
	int ii;					//Contador
	individuo_t ind_aux;	//Individuo auxiliar
	bool repetido;			//Booleano utilizado para identificar a un individuo repetido

	//Generacion aleatoria de individuos no repetidos
	for (ii = 0; ii < poblacion->cant_individuos; ii++){
		do{
			//Se asume en principio que el individuo no esta repetido
			repetido = false;

			//Generacion aleatoria de un cromosoma para el individuo
			generar_genotipo(&ind_aux);

			//El individuo esta ya presente, por lo que se repite el ciclo
			if (buscar_individuo(poblacion, &ind_aux, 0, ii - 1))
				repetido = true;
			//Insercion del individuo en la poblacion
			else{
				//Asignacion de la posicion en la poblacion
				ind_aux.pos_pob = ii;

				//Determinacion del fenotipo del individuo
				decodificar_genotipo(&ind_aux);

				//Solicitud de evaluacion de aptitud del individuo
				solicitar_aptitud(&ind_aux);

				//Insercion
				insertar_individuo(poblacion, &ind_aux, ii);
			}
		}while (repetido);
	}

	//Calcular las aptitudes de la poblacion
	aptitudes_poblacion(poblacion);

	//Calculo de estadisticas de la poblacion
	estadisticas_poblacion(poblacion);

	//Registro de aptitudes
	vec_max[0] = poblacion->aptitud_max;
	vec_min[0] = poblacion->aptitud_min;
	vec_prom[0] = poblacion->aptitud_prom;

	//Renormalizacion de la poblacion
	escalar_poblacion(poblacion);

	//Ordenamiento de la poblacion
	qsort(poblacion->individuos, poblacion->cant_individuos, sizeof(individuo_t), cmp_individuo_t);
}

/**
 * Procedimiento que implementa el metodo de seleccion proporcional Muestreo Estocastico Universal, el cual
 * asigna una cantidad de hijos a cada individuo en proporcion a su aptitud
 *
 * Parametros:
 * 		poblacion_t *poblacion 		- Referencia a el registro que contiene a los individuos a seleccionar
 * 		int cant_hijos				- Cantidad de hijos que se asignaran
 *		datos_select_t *vec_datos	- Referencia al arreglo que almacenara el resultado
 */
void muestreo_estocastico_universal(poblacion_t *poblacion, int cant_hijos, datos_select_t *vec_datos){
	//Variables
	int ii;				//Contador
	double rnd;			//Valor aleatorio
	double sum;			//Suma acumulada de aptitudes
	double sum_pob;		//Suma total de las aptitudes de la poblacion

	//Inicializacion del vector de datos y calculo de la suma acumulada
	sum_pob = 0.0;
	for (ii = 0; ii < poblacion->cant_individuos; ii++){
		vec_datos[ii].indx = ii;
		vec_datos[ii].hijos = 0;
		sum_pob += poblacion->individuos[ii].aptitud;
	}

	//Inicializacion de la variable aleatoria dentro del intervalo [0, 1 / cant_hijos]
	rnd = random_uniforme() * (1.0 / cant_hijos);

	//Proceso de asignacion de hijos
	sum = 0.0;
	for (ii = 0; ii < poblacion->cant_individuos; ii++){
		sum += (poblacion->individuos[ii].aptitud / sum_pob);
		while (rnd <= sum){
			(vec_datos[ii].hijos)++;
			rnd += (1.0 / cant_hijos);
		}
	}
}

/**
 * Seleccion de dos padres a partir del vector con los datos referentes a la cantidad de hijos
 * asignados a cada individuo, el cual se actualizara siempre que se realice una nueva seleccion
 *
 * Parametros:
 * 		poblacion_t *poblacion 		- Referencia a la poblacion con los individuos de interes
 * 		individuo_t *padre1			- Referencia al individuo padre 1 a seleccionar
 * 		individuo_t *padre2			- Referencia al individuo padre 2 a seleccionar
 * 		datos_select_t *vec_datos	- Referencia al arreglo con los datos necesarios para determinar los candidatos para la seleccion
 * 		int *ult_padre				- Referencia al entero que especifica el ultimo padre del vector de datos con hijos por asignar
 */
void seleccion(poblacion_t *poblacion, individuo_t *padre1, individuo_t *padre2, datos_select_t *vec_datos, int *ult_padre){
	//Variables
	int ii;						//Contador
	int rnd_pos;				//Posicion aleatoria
	int padres_s[2];			//Posiciones de los padres seleccionados
	datos_select_t temp;		//Dato temporal a usar para el posible swap

	//Seleccion de ambos padres de forma aleatoria, actualizando el vector de datos luego del cambio
	for (ii = 0; ii < 2; ii++){
		if ((*ult_padre) > 0){
			//Posicion aleatoria en el intervalo [0, ult_padre]
			rnd_pos = random_intervalo(0, *ult_padre);

			//Indice en la poblacion del individuo seleccionado
			padres_s[ii] = vec_datos[rnd_pos].indx;

			//Modificacion del vector de datos
			(vec_datos[rnd_pos].hijos)--;
			if (vec_datos[rnd_pos].hijos == 0){
				temp = vec_datos[rnd_pos];
				vec_datos[rnd_pos] = vec_datos[*ult_padre];
				vec_datos[*ult_padre] = temp;
				(*ult_padre)--;
			}
		}
		else{
			padres_s[ii] = vec_datos[0].indx;
			(vec_datos[0].hijos)--;
		}
	}

	//Asignacion de padres
	*padre1 = poblacion->individuos[padres_s[0]];
	*padre2 = poblacion->individuos[padres_s[1]];
}


/**
 * Creacion de una nueva generacion a partir de una generacion actual, a la cual se debera someter
 * a los efectos de los operadores clasicos de seleccion, cruce y mutacion. Adicionalmente, se utilizara
 * el esquema elitista del gap generacional
 *
 * Parametros:
 * 		poblacion_t *poblacion_act		- Referencia a la poblacion de la generacion actual y la cual sera sometida a cambios
 * 		int cant_reemplazo				- Entero con la cantidad de individuos a reemplazar en la generacion actual
 */
void generar_nueva_poblacion(poblacion_t *poblacion_act, int cant_reemplazo){
	//Variables
	int ii;														//Contador
	individuo_t	padre1, padre2;									//Padres seleccionados
	individuo_t hijo1, hijo2;									//Hijos generados
	datos_select_t vec_datos[poblacion_act->cant_individuos];	//Arreglo a utilizar para el proceso de seleccion
	int ult_padre;												//Posicion del ultimo padre con hijos en el vector auxiliar
	poblacion_t poblacion_des;									//Poblacion con los descendientes de la generacion actual

	//Asignacion de memoria a la poblacion de descendientes
	crear_poblacion(&poblacion_des, cant_reemplazo);

	//Definicion de la generacion en poblacion con los descendientes
	poblacion_des.generacion = poblacion_act->generacion + 1;

	//Asignacion de la cantidad de hijos correspondientes a cada individuo de la actual generacion
	muestreo_estocastico_universal(poblacion_act, cant_reemplazo, vec_datos);

	//Ordenamiento de forma descendente por cantidad de hijos asignados
	qsort(vec_datos, poblacion_act->cant_individuos, sizeof(datos_select_t), cmp_datos_select_t);

	//Determinacion de la posicion del ultimo individuo con hijos asignados
	for (ii = 0; (ii < poblacion_act->cant_individuos) && (vec_datos[ii].hijos > 0); ii++)
		;
	ult_padre = ii - 1;

	//Especificacion de los hijos asignados a cada individuo (Para estadisticas)
	for (ii = 0; ii < poblacion_act->cant_individuos; ii++)
		poblacion_act->individuos[vec_datos[ii].indx].cant_hijos = vec_datos[ii].hijos;

	//Generacion de los nuevos descendientes
	for (ii = 0; ii < cant_reemplazo; ii += 2){
		//Seleccion aleatoria de dos padres
		seleccion(poblacion_act, &padre1, &padre2, vec_datos, &ult_padre);

		//Cruce de los padres para generar una descendencia
		cruce(&padre1, &padre2, &hijo1, &hijo2);

		//Mutacion de cada hijo
		mutacion(&hijo1);
		mutacion(&hijo2);

		//Proceso de decodificacion de los individuos
		decodificar_genotipo(&hijo1);
		decodificar_genotipo(&hijo2);

		//Asignacion de las posiciones en la poblacion
		hijo1.pos_pob = ii;
		hijo2.pos_pob = ii + 1;

		//Solicitud de evaluacion de aptitud del par de descendientes
		solicitar_aptitud(&hijo1);
		solicitar_aptitud(&hijo2);

		//Insercion de los hijos
		insertar_individuo(&poblacion_des, &hijo1, ii);
		insertar_individuo(&poblacion_des, &hijo2, ii + 1);
	}

	//Calcular las aptitudes de la poblacion
	aptitudes_poblacion(&poblacion_des);

	//Reemplazo de la poblacion actual por un porcentaje (gap) de la descendencia
	reemplazar_poblacion(poblacion_act, &poblacion_des, cant_reemplazo);

	//Registro de aptitudes
	vec_max[poblacion_act->generacion] = poblacion_act->aptitud_max;
	vec_min[poblacion_act->generacion] = poblacion_act->aptitud_min;
	vec_prom[poblacion_act->generacion] = poblacion_act->aptitud_prom;

	//Libeacion de memoria de la poblacion de descendientes
	liberar_poblacion(&poblacion_des);
}

/**
 * Reemplazo de una cantidad especificada de los individuos de la poblacion actual por los de la poblacion de descendientes.
 * La cantidad de individuos a reemplazar se especifica por el parametro cant_reemplazo.
 *
 * Parametros:
 * 		poblacion_t *poblacion_obj 		- Referencia a el registro de la poblacion objetivo del reemplazo
 * 		poblacion_t *poblacion_src 		- Referencia a el registro de la poblacion fuente del reemplazo
 * 		int cant_reemplazo				- Entero con la cantidad de individuos a reemplazar en la poblacion objetivo
 */
void reemplazar_poblacion(poblacion_t *poblacion_obj, poblacion_t *poblacion_src, int cant_reemplazo){
	//Variables
	int ii;				//Contador
	
	//Definicion de la generacion en poblacion objetivo
	poblacion_obj->generacion = poblacion_src->generacion;

	//Se reemplazan los peores individuos dentro del porcentaje del gap por la descendencia
	for (ii = 0; ii < cant_reemplazo; ii++)
		poblacion_obj->individuos[ii] = poblacion_src->individuos[ii];

	//Calculo de estadisticas de la poblacion nueva
	estadisticas_poblacion(poblacion_obj);

	//Renormalizacion de la poblacion nueva
	escalar_poblacion(poblacion_obj);

	//Ordenamiento de la poblacion generada luego del reemplazo
	qsort(poblacion_obj->individuos, poblacion_obj->cant_individuos, sizeof(individuo_t), cmp_individuo_t);
}

/**
 * Calculo de las aptitudes de todos los individuos de la poblacion
 *
 * Parametros:
 * 		poblacion_t *poblacion 		- Referencia a la poblacion que se desea normalizar
 */
void aptitudes_poblacion(poblacion_t *poblacion){
	//Variables
	int ii;			//Contador

	//Calculo de aptitudes
	for (ii = 0; ii < poblacion->cant_individuos; ii++)
		calcular_aptitud(&(poblacion->individuos[ii]));
}

/**
 * Renormalizacion de toda la poblacion utilizando el metodo de renormalizacion lineal
 *
 * Parametros:
 * 		poblacion_t *poblacion 		- Referencia a la poblacion que se desea normalizar
 */
void escalar_poblacion(poblacion_t *poblacion){
	//Variables
	int ii;			//Contador
	double m;		//Pendiente de la recta de normalizacion
	double b;		//Interseccion en y de la recta de normalizacion

	//Calculo de m y b
	calcular_escala(poblacion->aptitud_min, poblacion->aptitud_max, poblacion->aptitud_prom, &m, &b);

	//Normalizacion de aptitudes
	for (ii = 0; ii < poblacion->cant_individuos; ii++)
		poblacion->individuos[ii].aptitud = escalar_aptitud(poblacion->individuos[ii].aptitud_orig, m, b);
}

/**
 * Procedimiento principal en donde se ejecuta el algoritmo genetico paralelo para la entonacion
 * de parametros de una red de retropogacion del error
 *
 * Parametros:
 * 		poblacion_t *poblacion 		- Referencia a el registro que almacenara los datos de la ultima poblacion generada
 */
void algoritmo_genetico(poblacion_t *poblacion){
	//Variables
	int ii;							//Contador
	poblacion_t poblacion_act;		//Poblacion de la actual generacion
	int cant_reemplazo;				//Cantidad de individuos que seran reemplazados entre generaciones
	time_t time_act;				//Tiempo actual

	//Generacion de una semilla aleatoria
	srand(time(NULL));

	//Determinacion de la cantidad de invididuos que seran reemplazados utilizando el gap generacional
	cant_reemplazo = (tam_poblacion * gap) + 0.5;
	if (cant_reemplazo % 2 != 0)
		cant_reemplazo++;

	//Creacion dinamica de las poblaciones a usar
	crear_poblacion(&poblacion_act, tam_poblacion);

	//Inicializacion de la poblacion
	iniciar_poblacion(&poblacion_act);

	//Mecanismo principal de evolucion durante una duracion especifica de generaciones
	for (ii = 1; ii <= max_gen; ii++){
		//Creacion de una nueva generacion a partir de la que se tiene en la poblacion actual, tomando en
		//consideracion la dinamica evolutiva propuesta y el gap generacional
		generar_nueva_poblacion(&poblacion_act, cant_reemplazo);

		//Se desean reportes y la iteracion actual es multiplo del valor proporcionado
		if ((gen_msj) && (iter_reporte > 0) && ((ii % iter_reporte) == 0 )){
			//Especificacion de la generacion
			fprintf(stdout, "\nReporte de la generacion %d\n", ii);

			//Impresion de la cantidad de segundos transcurridos desde el inicio del programa
			time(&time_act);
			fprintf(stdout, "\tTiempo transcurrido desde el inicio del programa: %.5lfseg\n", difftime(time_act, time_inicial));

			//Impresion de aptitudes
			fprintf(stdout, "\tAptitud maxima: %.5lf\n", poblacion_act.aptitud_max);
			fprintf(stdout, "\tAptitud minima: %.5lf\n", poblacion_act.aptitud_min);
			fprintf(stdout, "\tAptitud promedio: %.5lf\n", poblacion_act.aptitud_prom);

		}
	}

	//Copiado de la poblacion actual en la poblacion resultante
	reemplazar_poblacion(poblacion, &poblacion_act, tam_poblacion);

	//Liberacion de la memoria ocupada por las poblaciones locales
	liberar_poblacion(&poblacion_act);
}

/**
 * Impresion de datos de la poblacion proporcionada por el archivo de salida proporcionado
 *
 * Parametros:
 * 		poblacion_t *poblacion 		- Referencia a la poblacion a imprimir datos
 * 		FILE *out					- Archivo donde se mostrara la salida
 */
void reporte_estadisticas(poblacion_t *poblacion, FILE *out){
	//Variables
	int ii;										//Contador
	char str_bin[LONG_CROMOSOMA + 1];			//Cadena binaria
	time_t time_act;							//Tiempo de ejecucion actual

	//Encabezado
	fprintf(out, "UNIVERSIDAD DE CARABOBO\n");
	fprintf(out, "FACULTAD EXPERIMENTAL DE CIENCIAS Y TECNOLOGIA\n");
	fprintf(out, "ALGORITMO GENETICO PARALELO OPTIMIZADOR\n\n");
	fprintf(out, "					      TRABAJO ESPECIAL DE GRADO\n\n");
	fprintf(out, " \t\t'Algoritmo Genetico Paralelo para la Entonacion de Parametros de una\n");
	fprintf(out, " \t\t       Red Neuronal Artificial de Retropropagación del Error'\n\n");
	fprintf(out, "AUTOR: ORLANDO CASTILLO\n\n");
	fprintf(out, "                 REPORTE: ESTADISTICAS DE LA ULTIMA POBLACION\n\n");

	//Datos generales
	time(&time_act);
	fprintf(out, "\n\n\tDATOS GENERALES\n\n");
	fprintf(out, "Tiempo de ejecucion aproximado        = %.2lfseg\n", difftime(time_act, time_inicial));
	fprintf(out, "Total de generaciones                 = %d\n", poblacion->generacion);
	fprintf(out, "Cantidad de individuos                = %d\n", poblacion->cant_individuos);
	fprintf(out, "Probabilidad de cruce                 = %.3lf\n", pc);
	fprintf(out, "Probabilidad de mutacion              = %.3lf\n", pm);
	fprintf(out, "Porcentaje del gap generacional       = %.3lf\n\n", gap);

	fprintf(out, "Numero de iteraciones base para el entrenamiento                      = %d\n", iter_base);
	fprintf(out, "Valor de la potencia de diez a multiplicar el factor de iteraciones   = %d\n", pot_diez);
	fprintf(out, "Cantidad de repeticiones de entrenamiento e interrogatorio            = %d\n", cant_rep_red);
	switch(esquema_ejecucion){
		case MPI:
			fprintf(out, "Esquema de ejecucion seleccionado                                     = MPI\n");
			break;
		case FORK:
			fprintf(out, "Esquema de ejecucion seleccionado                                     = Fork\n");
			break;
		default:
			fprintf(out, "Esquema de ejecucion seleccionado                                     = Secuencial\n");
			break;
	}
	fprintf(out, "Nombre del archivo con los patrones de entrenamiento                  = %s\n", fnombre);

	fprintf(out, "\nReportes generados por el sistema: (Los asteriscos indican numeros enteros en el intervalo [1, %d]).\n", cant_rep_red);
	fprintf(out, "\t - ultima_generacion.out: Datos de la ultima generacion del algoritmo genetico.\n");
	fprintf(out, "\t - aptitudes.out: Aptitudes obtenidas por cada generacion del algoritmo genetico.\n");
	fprintf(out, "\t - %s_g.red: Estadisticas generales obtenidas luego de probar la mejor red obtenida.\n", fnombre);
	fprintf(out, "\t - %s_r*.red: Resultados del entrenamiento e interrogatorio * realizado a la mejor red obtenida.\n", fnombre);
	fprintf(out, "\t - %s_a*.red: Arquitectura de la mejor red generada durante la sesion de entrenamiento *.\n\n", fnombre);

	//Datos de individuos
	fprintf(out, "\n\n\tPOBLACION FINAL\n\n");
	fprintf(out, "La siguiente tabla muestra los datos de cada individuo de la ultima poblacion generada por el "
			"algoritmo genetico optimizador. La lista se encuentra ordenada de forma descendente respecto a la aptitud, "
			"por lo que el primer individuo representa la mejor red obtenida por el sistema y es a partir de la misma "
			"que se generan los reportes de red.\n\n");

	PRINT_LINE(ii, LONG_CROMOSOMA + 75, out);
	fprintf(out, "%-4s - %-*s - %-7s - %-35s\n", "Nro", LONG_CROMOSOMA, "Genotipo", "Aptitud",
			"Fenotipo (CC, CN1, CN2, APR, RAZ, MI, FACT_O, FACT_S) *");
	PRINT_LINE(ii, LONG_CROMOSOMA + 75, out);

	//Impresion de los datos de los individuos
	for (ii = poblacion->cant_individuos - 1; ii >= 0 ; ii--){
		conv_bin_str(poblacion->individuos[ii].genotipo.cromosoma, str_bin);
		fprintf(out, "%-4d   %s   %-7.5f", poblacion->cant_individuos - ii,str_bin, poblacion->individuos[ii].aptitud_orig);
		fprintf(out, "   (%d, %d, %d, %.5lf, %.3lf, %d, %d, %d) \n", poblacion->individuos[ii].fenotipo.cant_cap,
				poblacion->individuos[ii].fenotipo.cant_neu1, poblacion->individuos[ii].fenotipo.cant_neu2,
				poblacion->individuos[ii].fenotipo.const_apr, poblacion->individuos[ii].fenotipo.raz_mom,
				poblacion->individuos[ii].fenotipo.max_iter, poblacion->individuos[ii].fenotipo.func_act_o,
				poblacion->individuos[ii].fenotipo.func_act_s);
	}

	//Pie
	PRINT_LINE(ii, LONG_CROMOSOMA + 75, out);
	fprintf(out, "Aptitud maxima    = %.5lf\n", poblacion->aptitud_max);
	fprintf(out, "Aptitud minima    = %.5lf\n", poblacion->aptitud_min);
	fprintf(out, "Aptitud promedio  = %.5lf\n\n", poblacion->aptitud_prom);

	fprintf(out, "\n\n*Detalles del Fenotipo\n");
	fprintf(out, "\tCC     = Cantidad de capas ocultas\n");
	fprintf(out, "\tCN1    = Cantidad de neuronas en la capa oculta 1\n");
	fprintf(out, "\tCN2    = Cantidad de neuronas en la capa oculta 2\n");
	fprintf(out, "\tAPR    = Valor de la constante de aprendizaje\n");
	fprintf(out, "\tRAZ    = Valor de la razon de momentum\n");
	fprintf(out, "\tMI     = Maxima cantidad de iteraciones asignadas para el entrenamiento\n");
	fprintf(out, "\tFACT_O = Funcion de activacion utilizada por las neuronas de la capa oculta (1 - Sigmoide, "
			"2 - Tangente Hiperbolica)\n");
	fprintf(out, "\tFACT_S = Funcion de activacion utilizada por las neuronas de la capa de salida (1 - La misma "
			"que las neuronas ocultas, 2 - Lineal)\n");
}

/**
 * Impresion de las aptitudes minimas, maxima y promedio de cada generacion en un archivo proporcionado
 *
 * Parametros:
 * 		poblacion_t *poblacion 		- Referencia a la poblacion a imprimir datos
 * 		FILE *out					- Archivo donde se mostrara la salida
 */
void reporte_aptitudes(poblacion_t *poblacion, FILE *out){
	//Variables
	int ii;		//Contador

	//Encabezado
	fprintf(out, "UNIVERSIDAD DE CARABOBO\n");
	fprintf(out, "FACULTAD EXPERIMENTAL DE CIENCIAS Y TECNOLOGIA\n");
	fprintf(out, "ALGORITMO GENETICO PARALELO OPTIMIZADOR\n\n");
	fprintf(out, "					      TRABAJO ESPECIAL DE GRADO\n\n");
	fprintf(out, " \t\t'Algoritmo Genetico Paralelo para la Entonacion de Parametros de una\n");
	fprintf(out, " \t\t       Red Neuronal Artificial de Retropropagación del Error'\n\n");
	fprintf(out, "AUTOR: ORLANDO CASTILLO\n\n");

	//Impesion de aptitudes maximas, minimas y promedios
	fprintf(out, "                    REPORTE: APTITUDES DE CADA GENERACION\n\n");
	PRINT_LINE(ii, 49, out);
	fprintf(out, "%-10s - %-10s - %-10s - %-10s\n", "Generacion", "Apt. Max.", "Apt. Min.", "Apt. Prom.");
	PRINT_LINE(ii, 49, out);

	for (ii = 0; ii <= poblacion->generacion; ii++){
		fprintf(out, "%-10d   %-10.5lf   %-10.5lf   %-10.5lf\n", ii, vec_max[ii], vec_min[ii], vec_prom[ii]);
	}

	//Pie
	PRINT_LINE(ii, 49, out);
}

/**
 * Se instancia la mejor red de la poblacion en la modalidad de impresion de reportes
 *
 * Parametros:
 * 		poblacion_t *poblacion 		- Referencia a la poblacion donde se encuentra el mejor individuo
 */
void solicitar_reporte_mejor_individuo(poblacion_t *poblacion){
	//Variables
	individuo_t mejor_ind;		//Mejor individuo de la poblacion

	//Se asume ya ordenada la poblacion de forma ascendente
	mejor_ind = poblacion->individuos[poblacion->cant_individuos - 1];

	//El proceso no pudo crearse ejecutarse
	if (!instancia_rna_secuencial(comandoRNA, mejor_ind.fenotipo.cant_neu1, mejor_ind.fenotipo.cant_neu2,
			mejor_ind.fenotipo.const_apr, mejor_ind.fenotipo.raz_mom, mejor_ind.fenotipo.max_iter,
			mejor_ind.fenotipo.func_act_o, mejor_ind.fenotipo.func_act_s , cant_rep_red, 1, fruta, fnombre, sruta)){
		mostrar_error("Error al ejecutar la red con la mejor aptitud de la ultima generacion\n");
		finalizar_programa(-1);
	}
}

/**
 * Generacion aleatoria de un cromosoma para el individuo proporcionado
 *
 * Parametros:
 * 		individuo_t *ind 		- Referencia a la estructura que almacenara el cromosoma generado
 */
void generar_genotipo(individuo_t *ind){
	//Variables
	int ii;							//Contador

	//Generacion del cromosoma
	ind->genotipo.cromosoma = 0;
	for (ii = 0; ii < LONG_CROMOSOMA; ii++)
		if (flip(0.5))
			ind->genotipo.cromosoma += (UNO64_T << ii);
}

/**
 * Decodificacion del genotipo del individuo al fenotipo respectivo
 *
 * Parametros:
 * 		individuo_t *ind 		- Referencia a la estructura del individuo a decodificar
 */
void decodificar_genotipo(individuo_t *ind){
	//Variables
	uint64_t c_cromosoma;		//Copia del cromosoma del individuo

	//Se Copia el valor del cromosoma para posterior manipulacion
	c_cromosoma = ind->genotipo.cromosoma;

	//Cantidad de capas ocultas (Conversion binaria a entero sin signo)
	ind->fenotipo.cant_cap = (c_cromosoma & MASK_CANTCAP) + 1;

	//Cantidad de neuronas en la capa oculta 1 (Conversion binaria a entero sin signo)
	c_cromosoma >>= LONG_CANTCAP;
	ind->fenotipo.cant_neu1 = (c_cromosoma & MASK_CANTNEU1) + 1;

	//Cantidad de neuronas en la capa oculta 2 (en caso de tener) (Conversion binaria a entero sin signo)
	c_cromosoma >>= LONG_CANTNEU1;
	if (ind->fenotipo.cant_cap > 1)
		ind->fenotipo.cant_neu2 = (c_cromosoma & MASK_CANTNEU2) + 1;
	else
		ind->fenotipo.cant_neu2 = 0;

	//Valor de la constante de aprendizaje (Conversion binaria a un real en el intervalo [0,1] con 5 digitos de precision)
	c_cromosoma >>= LONG_CANTNEU2;
	ind->fenotipo.const_apr = (c_cromosoma & MASK_CONSTAPR) / (1.0 * MASK_CONSTAPR);

	//Valor de la razon de momentum (Conversion binaria a un real en el intervalo [0,1] con 3 digitos de precision)
	c_cromosoma >>= LONG_CONSTAPR;
	ind->fenotipo.raz_mom =  (c_cromosoma & MASK_RAZMOM) / (1.0 * MASK_RAZMOM);

	//Maxima cantidad de iteraciones para el entrenamiento (Conversion binaria a entero con signo)
	c_cromosoma >>= LONG_RAZMOM;
	ind->fenotipo.max_iter =  (c_cromosoma & MASK_FACTORITER) - (MASK_FACTORITER >> 1);
	ind->fenotipo.max_iter = iter_base + (ind->fenotipo.max_iter * pot_diez);

	//Funcion de activacion de las neuronas en las capas ocultas (Conversion binaria a entero sin signo)
	c_cromosoma >>= LONG_FACTORITER;
	ind->fenotipo.func_act_o =  (c_cromosoma & MASK_FUNCACT_O) + 1;

	//Funcion de activacion a utilizar (Conversion binaria a entero sin signo)
	c_cromosoma >>= LONG_FUNCACT_O;
	ind->fenotipo.func_act_s =  (c_cromosoma & MASK_FUNCACT_S) + 1;
}

/**
 * Cruce de un punto entre dos individuos padre, considerando una probabilidad de cruce pc
 *
 * Parametros:
 * 		individuo_t *padre1 		- Referencia a la estructura del padre 1
 * 		individuo_t *padre2 		- Referencia a la estructura del padre 2
 * 		individuo_t *hijo1 			- Referencia a la estructura del hijo 1
 * 		individuo_t *hijo2 			- Referencia a la estructura del hijo 2
 */
void cruce(individuo_t *padre1, individuo_t *padre2, individuo_t *hijo1, individuo_t *hijo2){
	//Variables
	uint64_t mascara;		//Mascara a utilizar para el proceso de cruce
	int punto;				//Punto de cruce

	//Inicializacion de datos
	hijo1->padre1 = padre1->genotipo;
	hijo1->padre2 = padre2->genotipo;
	hijo1->cruzo = false;

	hijo2->padre1 = padre1->genotipo;
	hijo2->padre2 = padre2->genotipo;
	hijo2->cruzo = false;

	//Se llevara a cabo el cruce si el evento aleatorio es exitoso
	if (flip(pc)){
		//Inicializacion
		hijo1->genotipo.cromosoma = 0;
		hijo2->genotipo.cromosoma = 0;

		//Determinacion del punto aleatorio de cruce dentro del intervalo entero [1, LONG_CROMOSOMA - 1]
		punto = random_intervalo(1, LONG_CROMOSOMA - 1);

		//Mascara que al hacer AND con un padre obtenemos la cola del mismo
		mascara = (UNO64_T << punto) - 1;

		//Intercambio de colas
		hijo1->genotipo.cromosoma = padre2->genotipo.cromosoma & mascara;
		hijo2->genotipo.cromosoma = padre1->genotipo.cromosoma & mascara;

		//Mascara que al hacer AND con un padres obtenemos la cabeza del mismo
		mascara = ~mascara;

		//Se les coloca la cabeza de los padres a sus hijos correspondientes
		hijo1->genotipo.cromosoma |= padre1->genotipo.cromosoma & mascara;
		hijo2->genotipo.cromosoma |= padre2->genotipo.cromosoma & mascara;

		//Registro de los datos del cruce
		hijo1->cruzo = true;
		hijo1->punto_cruce = punto;

		hijo2->cruzo = true;
		hijo2->punto_cruce = punto;

	}
	//Copia exacta de la informacion genetica a los dos hijos
	else{
		hijo1->genotipo = padre1->genotipo;
		hijo2->genotipo = padre2->genotipo;
	}
}


/**
 * Mutacion bit a bit del individuo proporcionado, considerando que cada bit puede ser mutado con una probabilidad pm
 *
 * Parametros:
 * 		individuo_t *individuo 			- Referencia a la estructura del indivduo a mutar
 */
void mutacion(individuo_t *individuo){
	//Variables
	int ii;				//Contador

	//Proceso de mutacion
	individuo->muto = false;
	for (ii = 0; ii < LONG_CROMOSOMA; ii++){
		//Se llevara a cabo la mutacion del bit ii si el evento aleatorio es exitoso
		if (flip(pm)){
			//Aplicacion del operador XOR. Si el bit es 0 la convierte a 1, y si el bit es 1 lo convierte a 0.
			individuo->genotipo.cromosoma ^= (UNO64_T << ii);
			individuo->muto = true;
		}
	}
}

/**
 * Ejecucion concurrente de la RNA con los parametros que el fenotipo del individuo representa.
 *
 * Parametros:
 * 		individuo_t *ind 		- Referencia a la estructura del individuo a evaluar
 */
void solicitar_aptitud(individuo_t *ind){
	//Solicitud de aptitude relativa al esquema de ejecucion
	switch(esquema_ejecucion){
		//Esquema con fork
		case FORK: {
			//Creacion de una nueva instancia de proceso (Verificando el exito)
			if(!instancia_rna_fork(comandoRNA, ind->fenotipo.cant_neu1, ind->fenotipo.cant_neu2, ind->fenotipo.const_apr,
					ind->fenotipo.raz_mom, ind->fenotipo.max_iter, ind->fenotipo.func_act_o,
					ind->fenotipo.func_act_s, cant_rep_red, fruta, fnombre, ind->pos_pob)){
				mostrar_error("Error al instanciar un proceso hijo usando fork\n");
				finalizar_programa(-1);
			}

			break;
		}
		//Esquema con mpi
		case MPI:{
			//Creacion de una nueva instancia de proceso (Verificando el exito)
			if (!instancia_rna_mpi(comandoRNA, ind->fenotipo.cant_neu1, ind->fenotipo.cant_neu2, ind->fenotipo.const_apr,
					ind->fenotipo.raz_mom, ind->fenotipo.max_iter, ind->fenotipo.func_act_o,
					ind->fenotipo.func_act_s, cant_rep_red, fruta, fnombre, ind->pos_pob)){
				mostrar_error("Error al instanciar un proceso hijo usando mpi\n");
				finalizar_programa(-1);
			}
			break;
		}
		//Esquema secuencial (NO RECOMENDADO)
		case SECUENCIAL: default:{
			break; //No se solicita nada, las aptitudes se conseguiran luego en la funcion calcular_aptitud
		}
	}
}


/**
 *
 * Calculo de la aptitud del individuo proporcionado. Para que este procedimiento pueda cumplir su objetivo,
 * depende de una llamada previa de 'solicitud_aptitud' con el indidividuo proporcionado.
 *
 * Parametros:
 * 		individuo_t *ind 		- Referencia a la estructura del individuo a evaluar
 */
void calcular_aptitud(individuo_t *ind){
	//Variables
	double ecm_prom;		//ECM promedio de la red representada por el individuo
	double norm_cn;			//Valor normalizado de la cantidad de neuronas ocultas
	double norm_iter;		//Valor normalizado de la cantidad de iteraciones maximas de entrenamiento
	double iter_max;		//Cantidad maxima de iteraciones
	double iter_min;		//Cantidad minima de iteraciones
	double costo_red;		//Valor de costo de la red

	//Calculo de la aptitud relativa al esquema de ejecucion
	switch(esquema_ejecucion){
		//Esquema con fork
		case FORK:{
			//Recepcion del ECM promedio
			if (!recibir_error_fork(ind->pos_pob, &ecm_prom))
				finalizar_programa(-1);
			break;
		}
		//Esquema con mpi
		case MPI:{
			//Recepcion del ECM promedio
			if (!recibir_error_mpi(ind->pos_pob, &ecm_prom))
				finalizar_programa(-1);
			break;
		}
		//Esquema secuencial
		case SECUENCIAL: default:{
			//Ejecucion de la RNA con los parametros proporcionados (Verificando el exito)
			if (!instancia_rna_secuencial( comandoRNA, ind->fenotipo.cant_neu1, ind->fenotipo.cant_neu2, ind->fenotipo.const_apr,
					ind->fenotipo.raz_mom, ind->fenotipo.max_iter, ind->fenotipo.func_act_o, ind->fenotipo.func_act_s,
					cant_rep_red, 0, fruta, fnombre, sruta)){
				mostrar_error("Error al ejecutar secuencialmente el programa de RNA\n");
				finalizar_programa(-1);
			}

			//Recepcion del ECM
			if (!recibir_error_secuencial(&ecm_prom))
				finalizar_programa(-1);
			break;
		}
	}

	//Calculo de la aptitud del individuo
	if (isnan(ecm_prom)){
		ind->aptitud = ind->aptitud_orig = 0.0;
	}
	else{
		//Valor normalizado de la cantidad de neuronas
		norm_cn = (ind->fenotipo.cant_neu1 + ind->fenotipo.cant_neu2) / ((double) (MASK_CANTNEU1 + MASK_CANTNEU2 + 2));

		//Valor normalizado de la cantidad maxima de iteraciones
		iter_max = iter_base + ((MASK_FACTORITER >> 1) * pot_diez);
		iter_min = iter_base - ((MASK_FACTORITER >> 1) * pot_diez);
		norm_iter = (ind->fenotipo.max_iter - iter_min) / ((double) (iter_max - iter_min));

		//Calculo del costo de la red, el cual depende de forma linea de tres criterios: ECM promedio,
		//cantidad de neuronas en las capas ocultas y cantidad de iteraciones
		costo_red = (3 * ecm_prom) + (2 * ecm_prom * norm_cn) + (ecm_prom * norm_iter);

		//Calculo de la aptitud
		ind->aptitud = 1.0 / (1.0 + costo_red);
		ind->aptitud_orig = ind->aptitud;
	}
}


	/**************************************************************/
	/***************   CUERPO PRINCIPAL   *************************/
	/**************************************************************/

int main(int argc, char *argv[]){
	//Variables
	char *error;				//Almacena un mensaje de error en caso de existir
	char *fnom;					//Nombre temporal del archivo que se desea abrir
	poblacion_t poblacion;		//Poblacion donde se almacenara el resultado del AG
	FILE *fgeneracion;			//Archivo para la impresion de los datos de la ultima generacion
	FILE *faptitudes;			//Archivo para la impresion de las aptitudes de cada generacion
	time_t time_fin;			//Tiempo de finalizacion del programa

	//Iniciar el reloj de ejecucion
	time(&time_inicial);

	//Verificacion de la especificacion correcta de parametros
	if ((error = parametros_entrada(argc, argv)) != NULL){
		mostrar_error(error);
		free(error);
		finalizar_programa(-1);
	}

	//Preparacion de los recursos asociados al esquema de ejecucion seleccionado
	preparar_esquema_ejecucion();

	//Apertura y validacion del archivo donde se mostraran los datos de la ultima generacion
	fnom = (char *) calloc(strlen(sruta) + 30, sizeof(char));
	sprintf(fnom, "%sultima_generacion.out",sruta);
	if ((fgeneracion = fopen(fnom, "w")) == NULL){
		mostrar_error("No pudo crearse el archivo 'ultima_generacion.out', en donde "
				"se mostraran los datos de la ultima generacion.\n");
		free(fnom);
		finalizar_programa(-1);
	}

	//Apertura y validacion del archivo donde se mostraran los datos de las aptitudes por cada generacion
	sprintf(fnom, "%saptitudes.out",sruta);
	if ((faptitudes = fopen(fnom, "w")) == NULL){
		mostrar_error("No pudo crearse el archivo 'aptitudes.out', en donde "
				"se mostraran las aptitudes maximas, minimas y promedios de cada generacion.\n");
		free(fnom);
		finalizar_programa(-1);
	}
	free(fnom);

	//Se se desean reportes, entonces se imprime un mensaje de inicio exitoso
	if (gen_msj){
		fprintf(stdout, "\nHa iniciado la ejecucion del algoritmo genetico optimizador...\n");
	}

	//Creacion dinamica de la poblacion que almacenara el resultado
	crear_poblacion(&poblacion, tam_poblacion);

	//Llamada al mecanismo de evolucion principal proporcionado por el algoritmo genetico, modificando a la variable
	//poblacion de forma tal que esta contenga los datos de la poblacion final obtenida
	algoritmo_genetico(&poblacion);

	//Generar reportes del mejor individuo
	solicitar_reporte_mejor_individuo(&poblacion);

	//Impresion de las aptitudes de cada generacion
	reporte_aptitudes(&poblacion, faptitudes);

	//Impresion de estadisticas de la ultima poblacion obtenida
	reporte_estadisticas(&poblacion, fgeneracion);

	//Liberacion de la memoria ocupada por la poblacion
	liberar_poblacion(&poblacion);

	//Clausura de archivos
	fclose(fgeneracion);
	fclose(faptitudes);

	//Se se desean reportes, entonces se imprime un mensaje de salida exitosa
	if (gen_msj){
		time(&time_fin);
		fprintf(stdout, "\nEl programa ha culminado exitosamente!\n");
		fprintf(stdout, "Tiempo estimado de ejecucion: %.5lfseg\n", difftime(time_fin, time_inicial));
		if (sruta != NULL)
			fprintf(stdout, "Puede proceder a revisar los reportes guardados en el directorio '%s'\n", sruta);
		else
			fprintf(stdout, "Puede proceder a revisar los reportes guardados en el directorio './'\n");
	}

	//FIN
	liberar_recursos();
	return 0;
}

	/************************************************************/
	/***************   PIE DEL PROGRAMA  ************************/
	/************************************************************/
