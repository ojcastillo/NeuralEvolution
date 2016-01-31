/***
	 Universidad de Carabobo
	 Facultad Experimental de Ciencias y Tecnologia

	 Trabajo Especial de Grado:
	 "Algoritmo Genético Paralelo para la Entonación de Parámetros de una
	 Red Neuronal Artificial de Retropropagación del Error."

	 Autor: 			Orlando Castillo

	 Tutor
	 Academico:			Joel Rivas

	 ------------------------------------------------------------------------------------------------
	 Archivo:			RN_BP.c

	 Descripcion:		Archivo principal del Algoritmo para la Red Neuronal Perceptron Generalizado
	 	 	 	 	 	con Retropropagación del Error.


	 Realizado por: 	Orlando Castillo

	 Version original:	Profs. Claudio Rocco y Jose Ali Moreno. (Universidad Central de Venezuela)
	 	 	 	 	 	Prof. Joel Rivas. (Universidad de Carabobo)
 ***/

/* Descripcion del programa:
 **
 ** El programa permite la construccion de una red perceptron generalizado
 ** con algoritmo de aprendizaje: retropropagacion del error
 ** que utiliza como regla de aprendizaje supervisado la regla delta
 ** generalizada.
 ** Pueden ser especificados por el usuario:
 ** Numero de unidades de entrada
 ** Numero de unidades de salida
 ** Numero de capas escondidas
 ** Numero de unidades en las capas escondidas
 **
 ** Luego de especificada la arquitectura, se lee el conjunto de patrones
 ** de entrenamiento, los cuales son normalizados al intervalo (0, 1).
 ** El aprendizaje procede utilizando dichos patrones.
 ** Para ello el usuario especifica:
 ** Constante de aprendizaje ETA
 ** Rata de momento ALFA
 ** Tolerancia maxima para el error
 ** Numero maximo de iteraciones
 **
 ** Al concluir el aprendizaje, los patrones de salida se desnormalizan y
 ** toda la informacion relativa a la arquitectura de la red, incluyendo
 ** los pesos y umbrales adaptados son almacenados en archivos de disco.
 **
 ** Reconstruyendo la red entrenada se pueden generar patrones de salida
 ** a partir de patrones de entrada leidos de archivos.
 **
 ** Muestras de patrones de entrenamiento y muestras de patrones adicionales
 ** se almacenan en archivos para su procesamiento ulterior.
 */

/*****************************************************************/
/***************   INSTRUCCIONES DE PREPROCESADOR  ***************/
/*****************************************************************/

/** Librerias **/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>
#include "../include/libparallel.h"

//Archivo con constantes de la compilacion
#ifdef HAVE_CONFIG_H
	#include <config.h>
#endif

/* Definicion de constantes utilizadas en las funciones  */
#define	NMXUNID    	129  	/* Numero maximo de unidades por capa oculta */
#define	NMXUNI_S   	"129"  	/* Numero maximo de unidades por capa oculta como un string*/
#define	NMXHID      2  		/* Numero maximo de capas ocultas */
#define	SEXIT       3  		/* Salida exitosa */
#define	REINIC      2  		/* Reiniciar */
#define	MALEXIT     1  		/* Salida en falla */
#define	CONTCALC    0  		/* Continuar calculo */
#define MAXFUNC_O	2		/* Cantidad de tipos de funciones de activacion permitidas para las neuronas ocultas*/
#define MAXFUNC_S	2		/* Cantidad de tipos de funciones de activacion permitidas para las neuronas de salida*/
#define PORC_ENTR	80		/* Porcentaje de patrones asignados para el entrenamiento */
#define MIN_SIG		0.2		/* Minimo valor para los patrones de entrada luego de normalizar, considerando el uso de sigmoide */
#define MAX_SIG		0.8		/* Maximo valor para los patrones de entrada luego de normalizar, considerando el uso de sigmoide*/
#define MIN_TAN		-0.5	/* Minimo valor para los patrones de entrada luego de normalizar, considerando el uso de la tangente*/
#define MAX_TAN		0.5		/* Maximo valor para los patrones de entrada luego de normalizar, considerando el uso de la tangente*/

/* Constantes para facilitar la legibilidad del programa */
#define FALSE       0
#define TRUE        1
#define NOT         !
#define AND         &&
#define OR          ||

#define MIN_REAL    -HUGE_VAL
#define MAX_REAL    +HUGE_VAL
#define MIN(x,y)    ((x)<(y) ? (x) : (y))
#define MAX(x,y)    ((x)>(y) ? (x) : (y))
#define pow2(x)     ((x)*(x))
#define vabs(x)		((x)>=(0) ? (x) : ((-1)*(x)))

#define PRINT_LINE(k, n, file)	for ((k) = 0; (k) < n; (k)++){fputc('-',file);} fputc('\n',file)

/*****************************************************************/
/***************   Tipos y enumerados  ***************************/
/*****************************************************************/
/* Definicion del tipo booleano o logico */
typedef int BOOL;

/* Tipos de funciones de activacion */
typedef enum {SIGMOIDE = 1, TANGENTE = 2} func_oculta_t;

typedef enum {IGUAL = 1, LINEAL = 2} func_salida_t;

/****************************************************************/
/***************   VARIABLES GLOBALES   *************************/
/****************************************************************/

float eta; 									/* Constante de aprendizaje */
float alfa; 								/* Razon de momento */
int nunit[NMXHID + 2 + 1]; 					/* Numero de unidades por capa */
int nhlayer;								/* Cantidad de capas ocultas */
int ninput, nsample; 						/* Numero de patrones de entrada */
int ninattr, noutattr; 						/* Numero de rasgos de entrada y de salida resp. */
int tipo_fun_o;								/* Tipo de funcion de transferencia a utilizar por las neuronas en las capas ocultas*/
int tipo_fun_s;								/* Tipo de funcion de transferencia a utilizar por las neuronas en la capas de salida*/
float *wtptr[NMXHID + 1]; 					/* Pesos de la red */
float *outptr[NMXHID + 2]; 					/* Salida de las unidades en la red */
float *errptr[NMXHID + 2]; 					/* Error de cada unidad de la red */
float *delw[NMXHID + 1]; 					/* Delta de cada peso de la red */
float **target;								/* Patrones de salida deseados*/
float **outpt;								/* Salidas de la red */
float **input;								/* Patrones de entrada */
int *indxEntrenamiento;						/* Indices de los patrones a usar para el entrenamiento */
int cantEntrenamiento;						/* Cantidad de patrones en el entranamiento */
int *indxInterrogatorio;					/* Indices de los patrones a usar para el interrogatorio */
int cantInterrogatorio;						/* Cantidad de patrones en el interrogatorio */
float *ep; 									/* Error por patron */
float *arrmaxi, *arrmaxt; 					/* Maximos para normalizacion */
float *arrmini, *arrmint; 					/* Minimos para normalizacion */
int esquemaEjecucion;						/* Esquema paralelo con el cual se ejecutara el programa */
int idSegmento;								/* Indentificador del segmento de memoria compartida */
int posMem;									/* Posicion de la memoria compartida asignada al proceso (se asume que es unica) */
int cantRep;								/* Cantidad de repeticiones de entrenamiento e interrogatorio */
BOOL genReportes;							/* Determina si se desea la generacion de archivos con reportes mas detallados */
BOOL patronesIniciados = FALSE;				/* Determina si los patrones han sido iniciados */
BOOL parametrosIniciados = FALSE;			/* Determina si los parametros del algoritmo han sido iniciados */
BOOL redIniciada = FALSE;					/* Determina si las estructuras de la red han sido iniciadas */
BOOL patronesDiv = FALSE;					/* Indica si ya los recursos para la division de patrones fueron creados */
BOOL esquemaIniciado = FALSE;				/* Determina si el esquema de ejecucion ha sido iniciado */
BOOL modoInterrogatorio = FALSE;			/* Especifica si se ejecuta el programa en modo de solo interrogatorio */
BOOL debug = TRUE;							/* Indica si se imprimen mensajes de error por stderr */
BOOL genMsj = TRUE;							/* Indica si se desean reportes de estado del sistema por salida stdout*/
time_t time_inicial, time_fin;				/* Tiempo de inicio de la ejecucion del programa*/
int iter_rep = 0;							/* Cantidad de iteraciones a transcurrir antes de generar un reporte de datos */

double errEntrenamientoNorm; 				/* Error total del entrenamiento */
double ecmEntrPromNorm;						/* Error cuadratico medio (ECM) de entrenamiento promedio alcanzado por la red*/
double errInterrogatorioNorm;				/* Error total del interrogatorio */
double ecmPredPromNorm;						/* Error cuadratico medio (ECM) de prediccion promedio normalizado alcanzado por el sistema */
double *corrMultEntr;						/* Factores de Correlacion Multiple alcanzados durante el entrenamiento */
double *corrMultInt;						/* Factores de Correlacion Multiple alcanzados durante el interrogatorio */
double *corrMultEntrProm;					/* Correlacion multiple promedio de todos los entrenamientos */
double *corrMultIntProm;					/* Correlacion multiple promedio de todos los interrogatorios */
float maxe; 								/* Error maximo permitido al sistema */
float maxep; 								/* Error maximo permitido del patron */
int result;
long cnt, cnt_num; 							/* numero de iteraciones y de iteraciones maximas */
char *fruta, *fnombre, *fruta_a, *fnombre_a, *rutaSalida;

/*********************************************************************/
/***************   PROTOTIPOS DE FUNCIONES   *************************/
/*********************************************************************/

/****************   Funciones Principales   *****************/

char *parse_modo_i(int argc, char *argv[]);

char *parse_modo_ie(int argc, char *argv[]);

char *parametros_entrada(int argc, char *argv[]);

int cmp_int(const void *val1, const void *val2);

void permutar_vector(int vec[], int tam_vec);

void init_patrones();

void init_red(void);

void initwt(void);

void definir_division(int porc_entrenamiento);

void liberar_division();

void asignar_patrones();

void liberar_patrones();

void liberar_parametros(void);

void liberar_red(void);

double normalizar_rasgo(double rasgo, double max_rasgo, double min_rasgo);

double desnormalizar_rasgo(double r_norm, double max_rasgo, double min_rasgo);

void normalizar(void);

void desnormalizar();

int random_intervalo(int a, int b);

double activacion_neurona_oculta(double net);

double derivada_neurona_oculta(double out);

double activacion_neurona_salida(double net);

double derivada_neurona_salida(double out);

double calcular_ecm(int *vecIndx, int cantIndx);

void calcular_correlacion(int *vecIndx, int cantIndx, double *corrM);

void forward(int i);

int introspective();

int rumelhart();

int entrenamiento(void);

void interrogatorio(void);

void reporte_global_corridas();

void reporte_entrenamiento_interrogatorio(int corrida);

void cargar_red();

void reporte_interrogatorio();

void reporte_arquitectura(int corrida);

void preparar_esquema_ejecucion();

void liberar_esquema_ejecucion();

void generar_resultados();

void liberar_recursos();

void finalizar_programa(int estado);

void mostrar_error(const char *msj);

/*************************************************************************/
/***************   IMPLEMENTACION DE FUNCIONES   *************************/
/*************************************************************************/

/****************   Funciones Principales   *****************/

/**
 * Obtiene los parametros necesarios para el modo de interrogatorio
 *
 * Parametros:
 * 		int argc 		- Cantidad de elementos en el arreglo argv
 * 		char *argv[] 	- Arreglo de cadenas de caracteres con la informacion de los parametros proporcionados por terminal
 */
char *parse_modo_i(int argc, char *argv[]){
	//Variables
	int ii;					//Contador
	BOOL valido = TRUE;		//Validez de la entrada
	char *msj = NULL;		//Mensaje de error a retornar si es necesario
	int opc = 0;			//Opcion de error
	char *err_pos[] = 	{
						//
						//Opcion 0
						//
						"Si desea la ejecucion en modo interrogatorio, debe especificar los\n"
						"siguientes 4 parametros obligatorios:\n"
						"   1.- Ruta al directorio con el archivo de patrones de entrada\n"
						"   2.- Nombre del archivo con los patrones de entrada (se asume\n"
						"       que tiene sufijo .dat)\n"
						"   3.- Ruta al directorio con el archivo de arquitectura de red\n"
						"   4.- Nombre del archivo con la arquitectura de red\n"
						"       (se asume que tiene sufijo .red)\n"
						"\n"
						"Puede tambien especificar el siguiente parametro opcional:\n"
						"\n"
						"   5.- Ruta al directorio en donde se guardaran los reportes resultantes\n"
						"\n"
						"Para mayor informacion, ejecute el programa con la opcion --help (-h)\n",
	};		//Mensajes de error posibles

	if (argc < 5 || argc > 6){
		valido = FALSE;
		opc = 0;
	}

	//Inicializacion
	fruta = NULL;
	fruta_a = NULL;
	fnombre = NULL;
	fnombre_a = NULL;
	rutaSalida = NULL;

	//Reocorrido del vector de parametros, validando el formato de los mismos
	for (ii = 1; ii <= 4 && valido; ii++){
		switch(ii){
			//Ruta al directorio con el archivo de patrones de entrada
			case 1:{
				fruta = (char *) calloc(strlen(argv[ii]) + 5, sizeof(char));
				strcpy(fruta, argv[ii]);
				break;
			}
			//Nombre del archivo con los patrones de entrada
			case 2:{
				fnombre = (char *) calloc(strlen(argv[ii]) + 5, sizeof(char));
				strcpy(fnombre, argv[ii]);
				break;
			}
			//Ruta al directorio con el archivo de arquitectura de red
			case 3:{
				fruta_a = (char *) calloc(strlen(argv[ii]) + 5, sizeof(char));
				strcpy(fruta_a, argv[ii]);
				break;
			}
			//Nombre del archivo con la arquitectura de red
			case 4:{
				fnombre_a = (char *) calloc(strlen(argv[ii]) + 5, sizeof(char));
				strcpy(fnombre_a, argv[ii]);
				break;
			}
		}
	}

	//Parametro opcional con la ruta del directorio en donde se almacenaran los resultados
	if (argc == 6 && valido){
		rutaSalida = (char *) calloc(strlen(argv[5]) + 5, sizeof(char));
		strcpy(rutaSalida, argv[5]);
	}

	//Se presento un error en los datos porporcionados
	if (!valido){
		msj = (char *) malloc(strlen(err_pos[opc]) + 21);
		strcpy(msj, err_pos[opc]);
	}
	//Se indica que se entra en modo interrogatorio
	else
		modoInterrogatorio = TRUE;

	//Mensaje resultante
	return msj;
}

/**
 * Obtiene los parametros necesarios para el modo con entrenamiento e interrogatorio
 *
 * Parametros:
 * 		int argc 		- Cantidad de elementos en el arreglo argv
 * 		char *argv[] 	- Arreglo de cadenas de caracteres con la informacion de los parametros proporcionados por terminal
 */
char *parse_modo_ie(int argc, char *argv[]){
	//Variables
	int ii;					//Contador
	BOOL valido = TRUE;		//Validez de la entrada
	char *msj = NULL;		//Mensaje de error a retornar si es necesario
	double temp_r;			//Variable real temporal
	int temp_i;				//Variable entera temporal
	int opc = 0;			//Opcion de error
	char *err_pos[] = 	{
						//
						//Opcion 0
						//
						"Para ejecutar el programa de Red Neuronal Artificial de Retropropagacion\n"
						"del Error en modo entrenamiento e interrogatorio, debe\n"
						"especificar los siguientes 11 parametros de forma obligatoria:\n"
						"\n"
						"   1.- Cantidad de neuronas en la capa oculta 1\n"
						"   2.- Cantidad de neuronas en la capa oculta 2\n"
						"   3.- Constante de aprendizaje\n"
						"   4.- Razon de momentum\n"
						"   5.- Maxima cantidad de iteraciones\n"
						"   6.- Funcion de activacion a utilizar por las neuronas en las\n"
						"       capas ocultas\n"
						"   7.- Funcion de activacion a utilizar por las neuronas en la\n"
						"       capa de salida\n"
						"   8.- Cantidad de repeticiones de entrenamiento e interrogatorio\n"
						"   9.- Ruta al directorio con el archivo de patrones de entrada\n"
						"   10.- Nombre del archivo con los patrones de entrada (se asume\n"
						"        que tiene sufijo .dat)\n"
						"   11.- Tipo del esquema de ejecucion del programa\n"
						"\n"
						"En caso de solicitar el esquema de ejecucion 2 (fork), debera especificar\n"
						"adicionalmente los siguientes 2 parametros de forma obligatoria:\n"
						"\n"
							"   12.- Identificador del segmento de memoria compartida\n"
							"   13.- Posicion asignada en la memoria compartida\n"
						"\n"
						"En caso de solicitar el esquema de ejecucion 0 (Ningun esquema), podra\n"
						"especificar los siguientes parametros opcionales:\n"
						"\n"
						"   12.- Ruta al directorio en donde se guardaran los reportes resultantes\n"
						"   13.- Generar reportes luego de transcurrida una cierta cantidad\n"
						"        de iteraciones de entrenamiento\n"
						"\n"
						"Para mayor informacion, ejecute el programa con la opcion --help (-h)\n",
						//
						//Opcion 1
						//
						"La cantidad de neuronas en la capa oculta 1 debe ser un valor entero\n"
						"en el intervalo [1, "NMXUNI_S"]\n",
						//
						//Opcion 2
						//
						"La cantidad de neuronas en la capa oculta 2 debe ser un valor entero\n"
						"en el intervalo [0, "NMXUNI_S"]\n",
						//
						//Opcion 3
						//
						"La constante de aprendizaje debe ser un valor real en el intervalo [0, 1]\n",
						//
						//Opcion 4
						//
						"La razon de momentum debe ser un valor real en el intervalo [0, 1]\n",
						//
						//Opcion 5
						//
						"La cantidad maxima de iteraciones debe ser un valor entero positivo\n",
						//
						//Opcion 6
						//
						"El tipo de funcion de activacion de las neuronas ocultas debe ser un\n"
						"valor entero en el intervalo [1, 2]\n",
						//
						//Opcion 7
						//
						"El tipo de funcion de activacion de las neuronas de salida debe ser\n"
						"un valor entero en el intervalo [1, 2]\n",
						//
						//Opcion 8
						//
						"La cantidad de repeticiones de entrenamiento e interrogatorio debe\n"
						"ser un valor entero no negativo\n",
						//
						//Opcion 9
						//
						"El valor del esquema de ejecucion a utilizar solo puede ser un entero\n"
						"en el intervalo [0, 3]\n",
						//
						//Opcion 10
						//
						"El identificador del segmento de memoria debe ser un entero no negativo\n",
						//
						//Opcion 11
						//
						"El valor de la posicion de memoria debe ser un entero no negativo\n",
						//
						//Opcion 12
						//
						"Si desea utilizar un esquema de ejecucion con fork, debe especificar 2\n"
						"parametros adicionales en el siguiente orden:\n"
						"  12.- Identificador del segmento de memoria compartida\n"
						"  13.- Posicion asignada en la memoria compartida\n",
						//
						//Opcion 13
						//
						"El valor de iteraciones para la generacion de reportes debe ser un\n"
						"entero no negativo\n"
					};		//Mensajes de error posibles

	if (argc < 12 || argc > 14){
		valido = FALSE;
		opc = 0;
	}

	//Valores invalidos que se utilizaran como banderas
	posMem = -1;
	idSegmento = -1;

	//Inicializacion
	fruta = NULL;
	fnombre = NULL;
	rutaSalida = NULL;

	//Reocorrido del vector de parametros, validando el formato de los mismos
	for (ii = 1; ii < 12 && valido; ii++){
		switch(ii){
			//Cantidad de unidades en la capa oculta 1
			case 1:{
				temp_i = atoi(argv[ii]);
				if (temp_i >= 1 && temp_i <= NMXUNID)
					nunit[1] = temp_i;
				else{
					valido = FALSE;
					opc = 1;
				}
				break;
			}
			//Cantidad de unidades en la capa oculta 2
			case 2:{
				temp_i = atoi(argv[ii]);
				if (temp_i >= 0 && temp_i <= NMXUNID){
					//Modificacion de la cantidad de capas ocultas
					if (temp_i > 0){
						nunit[2] = temp_i;
						nhlayer = 2;
					}
					else
						nhlayer = 1;
				}
				else{
					valido = FALSE;
					opc = 2;
				}
				break;
			}
			//Constante de aprendizaje
			case 3:{
				temp_r = strtod(argv[ii], NULL);
				if (temp_r >= 0.0 && temp_r <= 1.0)
					eta = temp_r;
				else{
					valido = FALSE;
					opc = 3;
				}
				break;
			}
			//Razon de momentum
			case 4:{
				temp_r = strtod(argv[ii], NULL);
				if (temp_r >= 0.0 && temp_r <= 1.0)
					alfa = temp_r;
				else{
					valido = FALSE;
					opc = 4;
				}
				break;
			}
			//Maxima cantidad de iteraciones
			case 5:{
				temp_i = atoi(argv[ii]);
				if (temp_i > 0)
					cnt_num = temp_i;
				else{
					valido = FALSE;
					opc = 5;
				}
				break;
			}
			//Tipo de funcion de activacion a utilizar por las neuronas en las capas ocultas
			//1.- Sigmoide
			//2.- Tangente Hiperbolica
			case 6:{
				temp_i = atoi(argv[ii]);
				if (temp_i >= 1 && temp_i <= MAXFUNC_O)
					tipo_fun_o = temp_i;
				else{
					valido = FALSE;
					opc = 6;
				}
				break;
			}
			//Tipo de funcion de activacion a utilizar por las neuronas en la capa de salida
			//1.- La misma que las neuronas ocultas
			//2.- Lineal
			case 7:{
				temp_i = atoi(argv[ii]);
				if (temp_i >= 1 && temp_i <= MAXFUNC_S)
					tipo_fun_s = temp_i;
				else{
					valido = FALSE;
					opc = 7;
				}
				break;
			}
			//Cantidad de repeticiones de entrenamiento e interrogatorio
			case 8:{
				temp_i = atoi(argv[ii]);
				if (temp_i >= 0)
					cantRep = temp_i;
				else{
					valido = FALSE;
					opc = 8;
				}
				break;
			}
			//Ruta al archivo con patrones de entrada y salida
			case 9:{
				fruta = (char *) calloc(strlen(argv[ii]) + 5, sizeof(char));
				strcpy(fruta, argv[ii]);
				break;
			}
			//Nombre del archivo con patrones de entrada y salida
			case 10:{
				fnombre = (char *) calloc(strlen(argv[ii]) + 5, sizeof(char));
				strcpy(fnombre, argv[ii]);
				break;
			}
			//Tipo de esquema de ejecucion
			case 11:{
				temp_i = atoi(argv[ii]);
				if (temp_i >= 0 && temp_i <= CANT_ESQUEMAS){
					switch(temp_i){
						//Esquema secuencial
						case 1:
							esquemaEjecucion = SECUENCIAL;
							genReportes = FALSE;
							break;
						//Esquema con fork
						case 2:
							esquemaEjecucion = FORK;
							genReportes = FALSE;
							break;
						case 3:
							esquemaEjecucion = MPI;
							genReportes = FALSE;
							break;
						//Sin esquema
						case 0: default:{
							genReportes = TRUE;
						}
					}
				}
				else{
					valido = FALSE;
					opc = 9;
				}
				break;
			}
		}
	}

	//Opciones adicional en caso de que se ejecute el programa en modo reporte
	if (genReportes){
		for (ii = 12; ii < argc && valido; ii++){
			switch (ii){
				//Ruta del directorio en donde se almacenran los reportes
				case 12:{
					rutaSalida = (char *) calloc(strlen(argv[12]) + 5, sizeof(char));
					strcpy(rutaSalida, argv[12]);
					break;
				}
				//Cantidad de iteraciones para generar reportes de entrenamiento
				case 13:{
					temp_i = atoi(argv[ii]);
					if (temp_i >= 0)
						iter_rep = temp_i;
					else{
						valido = FALSE;
						opc = 13;
					}
					break;
				}
			}
		}
	}
	//Opciones adicionales en caso de ejecutar el programa con esquema de fork
	else if (esquemaEjecucion == FORK){
		//Deben proporcionarse dos parametros adicionales de forma obligatoria
		if (argc == 14){
			for (ii = 12; ii < argc && valido; ii++){
				switch (ii){
					//Identificador del segmento de memoria compartida (solo en caso de usar un esquema de fork)
					case 12:{
						temp_i = atoi(argv[ii]);
						if (temp_i >= 0)
							idSegmento = temp_i;
						else{
							valido = FALSE;
							opc = 10;
						}
						break;
					}
					//Posicion de memoria asignada (solo en caso de usar un esquema de fork)
					case 13:{
						temp_i = atoi(argv[ii]);
						if (temp_i >= 0)
							posMem = temp_i;
						else{
							valido = FALSE;
							opc = 12;
						}
						break;
					}
				}
			}
		}
		else{
			valido = FALSE;
			opc = 12;
		}
	}

	//Se presento un error en los datos porporcionados
	if (!valido){
		msj = (char *) malloc(strlen(err_pos[opc]) + 21);
		strcpy(msj, err_pos[opc]);

		//Liberacion
		if (fnombre != NULL)
			free(fnombre);
		if (fruta != NULL)
			free(fruta);
	}

	//Mensaje resultante
	return msj;
}

/**
 * Asignacion de valores a los parametros especificados por la cadena de caracteres argv
 *
 * Parametros:
 * 		int argc 		- Cantidad de elementos en el arreglo argv
 * 		char *argv[] 	- Arreglo de cadenas de caracteres con la informacion de los parametros proporcionados por terminal
 */
char *parametros_entrada(int argc, char *argv[]){
	//Variables
	int ii;					//Contador
	BOOL valido = TRUE;		//Validez de la entrada
	char *msj = NULL;		//Mensaje de error a retornar si es necesario
	int temp_i;				//Variable entera temporal
	int opc = 0;			//Opcion de error
	char **vec_arg;			//Vector con los argumentos del programa
	int cant_arg;			//Cantidad de argumentos proporcionados
	char val_opc;			//Valor de una opcion especificada como argumento
	char *err_pos[] =
	{
		//
		//Opcion 0
		//
		"     RED NEURONAL DE RETROPROPAGACION DEL ERROR\n"
		"\n"
		"Para ejecutar el programa de Red Neuronal Artificial de Retropropagacion\n"
		"del Error, debe especificar como primer parametro el valor 0 si desea\n"
		"una ejecucion en modo solo interrogatorio o el valor 1 si desea\n"
		"realizar sesiones de entrenamiento e interrogatorio \n"
		"\n"
		"1) Modo interrogatorio\n"
		"\n"
		"Si desea la ejecucion en modo interrogatorio, debe especificar los\n"
		"siguientes 4 parametros obligatorios:\n"
		"   1.- Ruta al directorio con el archivo de patrones de entrada\n"
		"   2.- Nombre del archivo con los patrones de entrada (se asume\n"
		"       que tiene sufijo .dat)\n"
		"   3.- Ruta al directorio con el archivo de arquitectura de red\n"
		"   4.- Nombre del archivo con la arquitectura de red\n"
		"       (se asume que tiene sufijo .red)\n"
		"\n"
		"Puede tambien especificar el siguiente parametro opcional:\n"
		"\n"
		"   5.- Ruta al directorio en donde se guardaran los reportes resultantes\n"
		"\n"
		"2) Modo entrenamiento e interrogatorio\n"
		"\n"
		"Si desea la ejecucion en modo entrenamiento e interrogatorio, debe\n"
		"especificar los siguientes 11 parametros de forma obligatoria:\n"
		"   1.- Cantidad de neuronas en la capa oculta 1\n"
		"   2.- Cantidad de neuronas en la capa oculta 2\n"
		"   3.- Constante de aprendizaje\n"
		"   4.- Razon de momentum\n"
		"   5.- Maxima cantidad de iteraciones\n"
		"   6.- Funcion de activacion a utilizar por las neuronas en las\n"
		"       capas ocultas\n"
		"   7.- Funcion de activacion a utilizar por las neuronas en la\n"
		"       capa de salida\n"
		"   8.- Cantidad de repeticiones de entrenamiento e interrogatorio\n"
		"   9.- Ruta al directorio con el archivo de patrones de entrada\n"
		"   10.- Nombre del archivo con los patrones de entrada (se asume\n"
		"        que tiene sufijo .dat)\n"
		"   11.- Tipo del esquema de ejecucion del programa\n"
		"\n"
		"En caso de solicitar el esquema de ejecucion 2 (fork), debera especificar\n"
		"adicionalmente los siguientes 2 parametros de forma obligatoria:\n"
		"\n"
			"   12.- Identificador del segmento de memoria compartida\n"
			"   13.- Posicion asignada en la memoria compartida\n"
		"\n"
		"En caso de solicitar el esquema de ejecucion 0 (Ningun esquema), podra\n"
		"especificar los siguientes parametros opcionales:\n"
		"\n"
		"   12.- Ruta al directorio en donde se guardaran los reportes resultantes\n"
		"   13.- Generar reportes luego de transcurrida una cierta cantidad\n"
		"        de iteraciones de entrenamiento\n"
		"\n"
		"Para mayor informacion, ejecute el programa con la opcion --help (-h)\n",
		//
		//Opcion 1
		//
		"     RED NEURONAL DE RETROPROPAGACION DEL ERROR\n"
		"\n"
		"Modo de uso: RN_BP [opciones] [modo] [argumentos-modo]\n"
		"\n"
		"Programa que permite la construccion y/o uso de una Red Neuronal\n"
		"Artificial con algoritmo de aprendizaje: Retropropagacion del Error.\n"
		"El algoritmo utiliza la regla delta generalizada con la adicion\n"
		"de termino de momentum.\n"
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
		"  * MODOS DE EJECUCION:\n"
		"\n"
		"           0          Modo de interrogatorio.\n"
		"           1          Modo de entrenamiento e interrogatorio.\n"
		"\n"
		"  * ARGUMENTOS DE CADA MODO\n"
		"\n"
		"1) Modo interrogatorio\n"
		"\n"
		"Si desea la ejecucion en modo interrogatorio, debe especificar los siguientes\n"
		"4 parametros obligatorios:\n"
		"\n"
		"   1.- Ruta al directorio con el archivo de patrones de entrada\n"
		"\n"
		"           Ruta relativa o absoluta al directorio que contine\n"
		"           el archivo con los patrones de entrada a usar en\n"
		"           el interrogatorio. La ruta debe obligatoriamente\n"
		"           finalizar con el caracter delimitador de ruta\n"
		"           respectivo del sistema (Linux: '/')\n"
		"\n"
		"   2.- Nombre del archivo con los patrones de entrada.\n"
		"\n"
		"           Nombre del archivo sin sufijo de tipo con los\n"
		"           patrones de entrada a usar en el interrogatorio.\n"
		"           Se asume que el archivo se encuentra en el\n"
		"           directorio especificado en el argumento\n"
		"           anterior, es de tipo '.dat' y que cumple con\n"
		"           el formato requerido, el cual se especifica\n"
		"           en el manual proporcionado\n"
		"\n"
		"   3.- Ruta al directorio con el archivo  de arquitectura de red\n"
		"\n"
		"           Ruta relativa o absoluta al directorio que contine\n"
		"           el archivo con la arquitectura de la red que sera\n "
		"           interrogada. La ruta debe obligatoriamente\n"
		"           finalizar con el caracter delimitador de ruta\n"
		"           respectivo del sistema (Linux: '/')\n"
		"\n"
		"   4.- Nombre del archivo con la arquitectura de red\n"
		"\n"
		"           Nombre del archivo sin sufijo de tipo con los\n"
		"           parametros de la arquitectura de red que\n"
		"           se utilizara en el interrogatorio. Se asume que\n"
		"           el archivo se encuentra en el directorio\n"
		"           especificado en el argumento anterior, es de\n"
		"           tipo '.red' y que cumple con el formato requerido,\n"
		"           el cual se especifica en el manual proporcionado.\n"
		"\n"
		"Puede tambien especificar el siguiente parametro opcional:\n"
		"\n"
		"   5.- Ruta al directorio en donde se guardaran los reportes resultantes\n"
		"\n"
		"           Ruta relativa o absoluta al directorio en donde se guardaran\n"
		"           los reportes del interrogatorio. La ruta debe\n"
		"           obligatoriamente finalizar con el caracter delimitador\n"
		"           de ruta respectivo del sistema (Linux: '/')\n"
		"           Default: Directorio actual (Linux: './')\n"
		"\n"
		"2) Modo entrenamiento e interrogatorio\n"
		"\n"
		"Si desea la ejecucion en modo entrenamiento e interrogatorio, debe\n"
		"especificar los siguientes 11 parametros de forma obligatoria:\n"
		"\n"
		"   1.- Cantidad de neuronas en la capa oculta 1\n"
		"\n"
		"           Valor entero de la cantidad exacta de neuronas\n"
		"           en la capa oculta 1 (Min: 1, Max: "NMXUNI_S")\n"
		"   2.- Cantidad de neuronas en la capa oculta 2\n"
		"\n"
		"           Valor entero de la cantidad exacta de neuronas\n"
		"           en la capa oculta 2 (Min: 0, Max: "NMXUNI_S")\n"
		"\n"
		"   3.- Constante de aprendizaje\n"
		"\n"
		"           Valor real de la constante de\n"
		"           aprendizaje (Min: 0.0, Max: 1.0)\n"
		"\n"
		"   4.- Razon de momentum\n"
		"\n"
		"           Valor real de la razon de\n"
		"           momentum (Min: 0.0, Max: 1.0)\n"
		"\n"
		"   5.- Maxima cantidad de iteraciones\n"
		"\n"
		"           Valor entero positivo de la cantidad maxima de\n"
		"           iteraciones por cada sesion de entrenamiento\n"
		"\n"
		"   6.- Funcion de activacion a utilizar por las neuronas en las\n"
		"       capas ocultas\n"
		"\n"
		"           Valor entero que indica el tipo de funcion de activacion\n"
		"           a utilizar por cada neurona de las capas ocultas. Las\n"
		"           opciones disponibles son las siguientes:\n"
		"\n"
		"           1       -->   Funcion Sigmoide\n"
		"\n"
		"           2       -->   Funcion Tangente Hiperbolica\n"
		"\n"
		"   7.- Funcion de activacion a utilizar por las neuronas en la\n"
		"       capa de salida\n"
		"\n"
		"           Valor entero que indica el tipo de funcion de activacion\n"
		"           a utilizar por cada neurona de la capa de salida. Las\n"
		"           opciones disponibles son las siguientes:\n"
		"\n"
		"           1       -->   Las misma funcion que las neuronas ocultas\n"
		"\n"
		"           2       -->   Funcion Lineal\n"
		"\n"
		"   8.- Cantidad de repeticiones de entrenamiento e interrogatorio\n"
		"\n"
		"           Valor entero no negativo que determina la cantidad de sesiones\n"
		"           de entrenamiento a la que sera sometida la red con los \n"
		"           parametros especificados. En caso de especificar el valor 0,\n"
		"           se procedera a realizar solamente el entrenamiento con los\n"
		"           patrones proporcionados\n"
		"\n"
		"   9.- Ruta al directorio con el archivo de patrones de entrada\n"
		"\n"
		"           Ruta relativa o absoluta al directorio que contine\n"
		"           el archivo con los patrones de entrada a usar en\n"
		"           el entrenamiento e interrogatorio. La ruta debe\n"
		"           obligatoriamente finalizar con el caracter delimitador\n"
		"           de ruta respectivo del sistema (Linux: '/')\n"
		"\n"
		"   10.- Nombre del archivo con los patrones de entrada \n"
		"\n"
		"           Nombre del archivo sin sufijo de tipo con los\n"
		"           patrones de entrada a usar en el entrenamiento e\n"
		"           interrogatorio. Se asume que el archivo se encuentra\n"
		"           en el directorio especificado en el argumento\n"
		"           anterior, es de tipo '.dat' y que cumple con\n"
		"           el formato requerido, el cual se especifica\n"
		"           en el manual proporcionado\n"
		"\n"
		"   11.- Tipo del esquema de ejecucion del programa\n"
		"\n"
		"           Indica el esquema de ejecucion bajo el cual se\n"
		"           ejecutara el programa. El interes es permitir que\n"
		"           el programa pueda ejecutarse bajo diversos paradigmas,\n"
		"           en especial el paralelo. Las opciones disponibles\n"
		"           son las siguientes:\n"
		"\n"
		"           0       -->   El programa se ejecutara sin considerar\n"
		"                         algun esquema de ejcucion. Con esta\n"
		"                         opcion, se asume que el programa se\n"
		"                         ejecutara de manera independiente, por\n"
		"                         lo que procedera a generar reportes\n"
		"                         con estadisticas\n"
		"\n"
		"           1       -->   Esquema de ejecucion secuencial. Con esta\n"
		"                         opcion, el programa no generara reporte alguno,\n"
		"                         tan solo se limitara a reportar la medida de\n"
		"                         error calculada en un archivo para luego\n"
		"                         ser procesada por un proceso de interes.\n"
		"\n"
		"           2       -->   Esquema de ejecucion con fork utilizando\n"
		"                         memoria compartida. Con esta opcion, el\n"
		"                         programa no generara reporte alguno, tan\n"
		"                         solo se limitara a escribir en una\n"
		"                         posicion de la memoria compartida la\n"
		"                         medida de error calculada\n"
		"\n"
		"           3       -->   Esquema de ejecucion con mpi. Con esta opcion\n"
		"                         , el programa no generara reporte alguno, tan\n"
		"                         solo se limitara a comunicar por medio de un\n"
		"                         mensaje la medida de error calculada. El\n"
		"                         mensaje se envia al proceso padre del cual\n"
		"                         este programa se asume instanciado.\n"
		"\n"
		"En caso de solicitar el esquema de ejecucion 2, debera especificar\n"
		"adicionalmente los siguientes 2 parametros de forma obligatoria:\n"
		"\n"
		"   12.- Identificador del segmento de memoria compartida\n"
		"\n"
		"           Valor entero positivo del identificador de la memoria\n"
		"           compartida en donde se escribira el error calculado."
		"\n"
		"   13.- Posicion asignada en la memoria compartida\n"
		"\n"
		"           Valor entero positivo de la posicion de la memoria\n"
		"           compartida en donde se escribira el error calculado.\n"
		"           Se asume que no se requiere sincronizar el acceso a la\n"
		"           posicion especificada.\n"
		"\n"
		"En caso de solicitar el esquema de ejecucion 0 (Ningun esquema), podra\n"
		"especificar los siguientes parametros opcionales:\n"
		"\n"
		"   12.- Ruta al directorio en donde se guardaran los reportes resultantes\n"
		"\n"
		"           Ruta relativa o absoluta al directorio en donde se guardaran\n"
		"           los reportes de estadisticas generados. La ruta debe\n"
		"           obligatoriamente finalizar con el caracter delimitador\n"
		"           de ruta respectivo del sistema (Linux: '/')\n"
		"           Default: Directorio actual (Linux: './')\n"
		"\n"
		"   13.- Generar reportes luego de transcurrida una cierta cantidad\n"
		"        de iteraciones de entrenamiento\n"
		"\n"
		"           Valor entero no negativo que le indica al programa cada\n"
		"           cuantas iteraciones de entrenamiento se desea un reporte\n"
		"           de datos. En caso de especificar 0,o si se proporciona\n"
		"           la opcion 'mensajes=n' al programa, no se imprimira ningun\n"
		"           reporte. Default: 0\n"
		"\n"
		"Para mayor informacion, revisar el manual de usuario distribuido con\n"
		"el software\n",
		//
		//Opcion 2
		//
		"RN_BP 1.0\n"
		"Red Neuronal Artificial con algoritmo de aprendizaje: Retropropagacion del Error.\n"
		"\n"
		"Desarrollado por: Orlando Castillo y Joel Rivas\n"
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
		"mayor informacion\n",
		//
		//Opcion 6
		//
		"El valor de la modalidad de ejecucion solo puede ser 0 o 1, donde 0\n"
		"indica una ejecucion de solo interrogatorio y 1 indica una ejecucion\n"
		"de sesiones de entrenamiento e interrogatorio.\n"
	};		//Mensajes de error posibles

	//No se especificaron parametros
	if (argc == 1){
		valido = FALSE;
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
					valido = FALSE;
					opc = 1;
				}
				else if (strcmp(argv[ii], "--version") == 0){
					valido = FALSE;
					opc = 2;
				}
				else if (strncmp(argv[ii], "--debug=", 8) == 0){
					if (strlen(argv[ii]) != 9){
						valido = FALSE;
						opc = 4;
					}
					else{
						val_opc = argv[ii][8];
						if (val_opc == 'y')
							debug = TRUE;
						else if (val_opc == 'n')
							debug = FALSE;
						else{
							valido = FALSE;
							opc = 4;
						}
					}
				}
				else if (strncmp(argv[ii], "--mensajes=", 11) == 0){
					if (strlen(argv[ii]) != 12){
						valido = FALSE;
						opc = 5;
					}
					else{
						val_opc = argv[ii][11];
						if (val_opc == 'y')
							genMsj = TRUE;
						else if (val_opc == 'n')
							genMsj = FALSE;
						else{
							valido = FALSE;
							opc = 5;
						}
					}
				}
				else{
					valido = FALSE;
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
			valido = FALSE;
			opc = 0;
		}

		//Sigue siendo valido
		if (valido){
			temp_i = atoi(vec_arg[0]);
			switch(temp_i){
				//Modo inteorrogatorio
				case 0:
					msj = parse_modo_i(cant_arg, vec_arg);
					break;
				//Modo entrenamiento e interrogatorio
				case 1:
					msj = parse_modo_ie(cant_arg, vec_arg);
					break;
				//Modo invalido
				default:{
					valido = FALSE;
					opc = 6;
					break;
				}
			}
		}

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
 * Funcion de comparacion para variables de tipo int. El principal proposito
 * es utilizar esta funcion como parametros a las funciones de libreria qsort
 *
 * Parametros:
 * 		const void *val1 		- Referencia generica al primer valor
 * 		const void *val1 		- Referencia generica al segundo valor
 */
int cmp_int(const void *val1, const void *val2){
	//Variables
	int *x, *y;		//Enteros a comparar

	//Casteo a los tipos correctos
	x = (int *) val1;
	y = (int *) val2;

	//Proceso de comparacion
	if (*x < *y)
		return -1;
	if (*x > *y)
		return 1;
	return 0;
}

/**
 * Permutacion aletoria de un vector proporcionado con ditribucion uniforme
 *
 * Parametros:
 * 		int vec[] 		- Vector a permutar
 * 		int tam_vec 	- Tamanio del vector
 */
void permutar_vector(int vec[], int tam_vec){
	//Variables
	int indx_rnd;	//Indice generado aleatoriamente
	int temp;		//Valor temporal para el intercambio
	int ii;			//Contador

	//Permutacion aleatoria del vector
	for (ii = 0; ii < (tam_vec - 1); ii++){
		indx_rnd = random_intervalo(ii, tam_vec - 1);
		temp = vec[indx_rnd];
		vec[indx_rnd] = vec[ii];
		vec[ii] = temp;
	}
}

/**
 * Lectura de los patrones de entrada y de salida de la red
 */
void init_patrones(){
	//Variables
	int ii, jj;				//Contadores
	char *file;				//Ruta y nombre del archivo con los patrones
	FILE *fpatrones;		//Archivo con los patrones de entrada y salida
	BOOL valido;			//Determina la validez del formato del archivo

	//Combinacion de la ruta y nombre del archivo
	file = (char *) calloc(strlen(fruta) + strlen(fnombre) + 10, sizeof(char));
	strcat(file, fruta);
	strcat(file, fnombre);
	strcat(file, ".dat");

	//Si la ruta resulta invalida, se reporta el error
	if ((fpatrones = fopen(file, "r")) == NULL){
		mostrar_error("No fue posible abrir el archivo de patrones '");
		mostrar_error(fnombre);
		mostrar_error("' exitosamente. Verifique que la ruta proporcionada al directorio '");
		mostrar_error(fruta);
		mostrar_error("' sea valida y que el archivo sea de tipo '.dat'\n");
		free(file);
		finalizar_programa(-1);
	}
	free(file);

	//Se asume que el formato del archivo es valido
	valido = TRUE;

	//Lectura de la cantidad de rasgos de entrada, cantidad de rasgos de salidas y cantidad de patrones
	if (fscanf(fpatrones, "%d %d %d", &ninattr, &noutattr, &ninput) == EOF)
		valido = FALSE;

	if (valido){
		//Asignacion de memoria para las entradas por patron
		input = (float **) calloc(ninput, sizeof(float *));
		for (ii = 0; ii < ninput; ii++)
			*(input + ii) = (float *) calloc(ninattr, sizeof(float));

		//Asignacion de memoria para las salidas deseadas por patron
		target = (float **) calloc(ninput, sizeof(float *));
		for (ii = 0; ii < ninput; ii++)
			*(target + ii) = (float *) calloc(noutattr, sizeof(float));

		//Asignacion de memoria para la salida del sistema por patron
		outpt = (float **) calloc(ninput, sizeof(float *));
		for (ii = 0; ii < ninput; ii++)
			*(outpt + ii) = (float *) calloc(noutattr, sizeof(float));

		//Memoria para las estructuras a usar en la normalizacion
		arrmaxi = (float *) calloc(ninattr, sizeof(float));
		arrmini = (float *) calloc(ninattr, sizeof(float));
		arrmaxt = (float *) calloc(noutattr, sizeof(float));
		arrmint = (float *) calloc(noutattr, sizeof(float));

		//Se registra que los recursos para los patrones han sido inicializados
		patronesIniciados = TRUE;

		//Lectura de los patrones
		for (ii = 0; ii < ninput && valido; ii++) {
			for (jj = 0; jj < ninattr && valido; jj++)
				if (fscanf(fpatrones, "%f", &input[ii][jj]) == EOF)
					valido = FALSE;
			for (jj = 0; jj < noutattr && valido; jj++)
				if (fscanf(fpatrones, "%f", &target[ii][jj]) == EOF)
					valido = FALSE;
		}
	}

	//El formato del archivo es incorrecto
	if (!valido){
		mostrar_error("El formato del archivo de patrones '");
		mostrar_error(fnombre);
		mostrar_error("' es invalido. Revise el manual de usuario para los detalles\n");
		finalizar_programa(-1);
	}

	//El archivo no pudo cerrarse de forma exitosa
	if ((ii = fclose(fpatrones)) != 0) {
		mostrar_error("El archivo con los patrones de entrada '");
		mostrar_error(fnombre);
		mostrar_error("' no pudo cerrarse exitosamente\n");
		finalizar_programa(-1);
	}
}

/* Disposicion de almacenaje dinamico para la red */
void init_red(void) {
	//Variables
	int i;

	//Inicializacion
	nunit[nhlayer + 2] = 0;

	//Especificacion de la cantidad de neuronas de entrada y salida
	nunit[0] = ninattr;
	nunit[nhlayer + 1] = noutattr;

	//Asignacion de memoria para la estructura que almacenara los errores por patrones
	ep =  (float *) calloc(ninput, sizeof(float));

	/* asignacion del resto de los apuntadores */
	for (i = 0; i < (nhlayer + 1); i++) { /* 3 capas (con la de entrada) implica 2 conj pesos */
		wtptr[i] = (float *) calloc((nunit[i] + 1) * (nunit[i + 1] + 1), sizeof(float));
		delw[i] = (float *) calloc((nunit[i] + 1) * (nunit[i + 1] + 1), sizeof(float));
	}

	for (i = 0; i < (nhlayer + 2); i++) {
		outptr[i] = (float *) calloc(nunit[i] + 1, sizeof(float));
		errptr[i] = (float *) calloc(nunit[i] + 1, sizeof(float));
	}

	/* asignacion de umbrales de salida */
	for (i = 0; i < nhlayer + 1; i++) {
		*(outptr[i] + nunit[i]) = -1.0; /*OJO*//* valor constante de los umbrales */
	}

	//Memoria para los factores de correlacion
	corrMultEntr = (double *) calloc(noutattr, sizeof(double));
	corrMultEntrProm = (double *) calloc(noutattr, sizeof(double));
	corrMultInt = (double *) calloc(noutattr, sizeof(double));
	corrMultIntProm = (double *) calloc(noutattr, sizeof(double));

	//Se registra que las estructuras de la red han si inicializadas
	redIniciada = TRUE;
} /* fin init */



/* Inicializar pesos con numeros aleatorios entre -0.3 y 0.3 */
void initwt(void) {
	int i, j;

	for (j = 0; j < (nhlayer + 1); j++) /* se mueve por capa */
		for (i = 0; i < ((nunit[j] + 1) * nunit[j + 1]); i++) {
			*(wtptr[j] + i) = 0.6 * (rand() / (1.0 * RAND_MAX)) - 0.3;
			*(delw[j] + i) = 0.0;
		}
} /* fin initwt */

/**
 * Creacion de las estructuras con los indices a patrones de acuerdo a los porcentajes proporcionados
 */
void definir_division(int porc_entrenamiento){
	//Memoria para los vectores de indices de patrones
	cantEntrenamiento = ninput * (porc_entrenamiento / 100.0);
	indxEntrenamiento = (int *) calloc(cantEntrenamiento + 2, sizeof(int));
	cantInterrogatorio = ninput - cantEntrenamiento;
	indxInterrogatorio = (int *) calloc(cantInterrogatorio + 2, sizeof(int));

	//Registro de la creacion de recursos
	patronesDiv = TRUE;
}

/**
 * Liberacion de los recursos creados para la division de indices de patrones
 */
void liberar_division(){
	//Memoria de los vectores de indices de patrones
	free(indxEntrenamiento);
	free(indxInterrogatorio);

	//Registro de la liberacion de recursos
	patronesDiv = FALSE;
}

/**
 * Asignacion aleatoria de indices de los patrones a utilizar tanto en el entrenamiento como interrogatorio
 *
 */
void asignar_patrones(){
	//Variables
	int ii;					//Contador
	int vec_aux[ninput];	//Vector auxiliar con los indices

	//Generacion de indices iniciales
	for (ii = 0; ii < ninput; ii++)
		vec_aux[ii] = ii;

	//Permutacion aleatoria del vector de indices
	permutar_vector(vec_aux, ninput);

	//Asignacion de los indices al entrenamiento
	for (ii = 0; ii < cantEntrenamiento; ii++)
		indxEntrenamiento[ii] = vec_aux[ii];

	//Asignacion de los indices al interrogatorio
	for (ii = cantEntrenamiento; ii < ninput; ii++)
		indxInterrogatorio[ii - cantEntrenamiento] = vec_aux[ii];
}


/**
 * Liberacion de la memoria asignada a las estrcturas que almacenan los patrones de entrada y salida
 */
void liberar_patrones(){
	//Variables
	int ii;		//Contador

	//Memoria de los patrones de entrada
	for (ii = 0; ii < ninput; ii++)
		free(*(input + ii));
	free(input);

	//Memoria de los patrones de salida
	for (ii = 0; ii < ninput; ii++)
		free(*(target + ii));
	free(target);

	//Memoria de la salida del sistema
	for (ii = 0; ii < ninput; ii++)
		free(*(outpt + ii));
	free(outpt);

	//Memoria de las estructuras para la normalizacion
	free(arrmaxi);
	free(arrmini);
	free(arrmaxt);
	free(arrmint);

	//Se registra la liberacion
	patronesIniciados = FALSE;
}

/**
 * Libreacion de la memoria asignada a los parametros del sistema
 */
void liberar_parametros(void){
	//Libreacion de la memoria asignada a el nombre y ruta del archivo
	free(fnombre);
	free(fruta);
	if (rutaSalida != NULL)
		free(rutaSalida);

	//Se registra la liberacion
	parametrosIniciados = FALSE;
}

/**
 * Libreacion de la memoria asignada a la arquitectura de la red
 */
void liberar_red(void){
	//Variables
	int m;		//Contador

	//Liberacion de memoria
	free(ep);
	for (m = 0; m < nhlayer + 2; m++){
		if (m < nhlayer + 1){
			free(*(wtptr + m));
			free(*(delw + m));
		}
		free(*(outptr + m));
		free(*(errptr + m));
	}
	free(corrMultEntr);
	free(corrMultEntrProm);
	free(corrMultInt);
	free(corrMultIntProm);

	//Se registra la liberacion
	redIniciada = FALSE;
}

/**
 * Normalizacion de un rasgo proporcionado considerando la funcion de activacion de las neuronas ocultas
 *
 * Parametros:
 * 		double rasgo 		- Valor del rasgo a normalizar
 * 		double max_rasgo	- Maximo valor del rasgo proporcionado
 * 		double min_rasgo	- Minimo valor del rasgo proporcionado
 */
double normalizar_rasgo(double rasgo, double max_rasgo, double min_rasgo){
	//Valor
	double norm;		//Valor normalizado

	switch(tipo_fun_o){
		//Sigmoide
		case SIGMOIDE:{
			norm = ((MAX_SIG - MIN_SIG) * ((rasgo - min_rasgo) / (max_rasgo - min_rasgo))) + MIN_SIG;
			break;
		}
		//Tangente hipoerbolica
		case TANGENTE: default: {
			norm = ((MAX_TAN - MIN_TAN) * ((rasgo - min_rasgo) / (max_rasgo - min_rasgo))) + MIN_TAN;
			break;
		}
	}

	//Resultado
	return norm;
}

/**
 * Desnormalizacion de un rasgo proporcionado considerando la funcion de activacion de las neuronas ocultas
 *
 * Parametros:
 * 		double r_norm 		- Valor del rasgo a desnormalizar
 * 		double max_rasgo	- Maximo valor del rasgo proporcionado
 * 		double min_rasgo	- Minimo valor del rasgo proporcionado
 */
double desnormalizar_rasgo(double r_norm, double max_rasgo, double min_rasgo){
	//Valor
	double rasgo;		//Valor desnormalizado

	switch(tipo_fun_o){
		//Sigmoide
		case SIGMOIDE:{
			rasgo = ((max_rasgo - min_rasgo) * ((r_norm - MIN_SIG) / (MAX_SIG - MIN_SIG))) + min_rasgo;
			break;
		}
		//Tangente hipoerbolica
		case TANGENTE: default: {
			rasgo = ((max_rasgo - min_rasgo) * ((r_norm - MIN_TAN) / (MAX_TAN - MIN_TAN))) + min_rasgo;
			break;
		}
	}

	//Resultado
	return rasgo;
}


/* 	 Funcion previa a la etapa de aprendizaje e interrogatorio de la red, que tiene */
/*   como objetivo normalizar la data de entrada 					 				*/
void normalizar(void) {
	int i, j;

	/* inicializacion de estructuras para el proceso de normalizacion */
	for (j = 0; j < ninattr; j++) {
		arrmaxi[j] = MIN_REAL;
		arrmini[j] = MAX_REAL;
	}

	for (j = 0; j < noutattr; j++) {
		arrmaxt[j] = MIN_REAL;
		arrmint[j] = MAX_REAL;
	}

	/* Normalizacion de los rasgos de entrada */
	for (j = 0; j < ninattr; j++) {
		/* Busqueda de minimos y maximos en los rasgos de entrada */
		for (i = 0; i < ninput; i++){
			//Maximo
			if (arrmaxi[j] < input[i][j])
				arrmaxi[j] = input[i][j];
			//Minimo
			if (arrmini[j] > input[i][j])
				arrmini[j] = input[i][j];
		}

		/* Normalizacion */
		for (i = 0; i < ninput; i++)
			input[i][j] = normalizar_rasgo(input[i][j], arrmaxi[j], arrmini[j]);

	}

	/* Normalizacion de la salida */
	for (j = 0; j < noutattr; j++) {
		/* Busqueda de minimos y maximos en los rasgos de salida */
		for (i = 0; i < ninput; i++){
			//Maximo
			if (arrmaxt[j] < target[i][j])
				arrmaxt[j] = target[i][j];
			//Minimo
			if (arrmint[j] > target[i][j])
				arrmint[j] = target[i][j];
		}

		/* Normalizacion */
		for (i = 0; i < ninput; i++)
			target[i][j] = normalizar_rasgo(target[i][j], arrmaxt[j], arrmint[j]);
	}
} /* fin normaliza_apr */

/* 	 Funcion posterior a la etapa de aprendizaje e interrogatorio de la red, que tiene  */
/*   como objetivo desnormalizar la data de entrada 									*/
void desnormalizar() {
	int i, j;

	//Desnormalizacion de los rasgos de entrada
	for (j = 0; j < ninattr; j++) {
		for (i = 0; i < ninput; i++)
			input[i][j] = desnormalizar_rasgo(input[i][j], arrmaxi[j], arrmini[j]);
	}

	//Desnormalizacion de los rasgos de salida
	for (j = 0; j < noutattr; j++) {
		for (i = 0; i < ninput; i++){
			target[i][j] = desnormalizar_rasgo(target[i][j], arrmaxt[j], arrmint[j]);
			outpt[i][j] = desnormalizar_rasgo(outpt[i][j], arrmaxt[j], arrmint[j]);
		}
	}
} /* fin desnormaliza */

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
 * Aplicacion de la funcion de activacion a una neurona oculta
 *
 * Parametros:
 * 		double net		- Suma ponderada de las activaciones de entrada
 */
double activacion_neurona_oculta(double net){
	//Variables
	double salida;		//Activacion generada

	switch (tipo_fun_o){
		/*Sigmoide*/
		case SIGMOIDE:{
			salida = 1.0 / (1.0 + exp(-net));
			break;
		}
		/*Tangente Hiperbolica*/
		case TANGENTE: default:{
			salida = tanh(net);
			break;
		}
	}

	//Resultado
	return salida;
}

/**
 * Evaluacion de la derivada con la salida obtenida por una neurona oculta
 *
 * Parametros:
 * 		double out		- Salida de la neurona oculta
 */
double derivada_neurona_oculta(double out){
	//Variables
	double derivada;		//Valor de la derivada evaluada en la salida proporcionada

	switch (tipo_fun_o) {
		/*Sigmoide*/
		case SIGMOIDE:{
			derivada = (1 - out) * out;
			break;
		}
		/*Tangente Hiperbolica*/
		case TANGENTE: default:{
			derivada = (1 - out) * (1 + out);
			break;
		}
	}

	//Resultado
	return derivada;
}

/**
 * Aplicacion de la funcion de activacion a una neurona de salida
 *
 * Parametros:
 * 		double net		- Suma ponderada de las activaciones de entrada
 */
double activacion_neurona_salida(double net){
	//Variables
	double salida;		//Activacion generada

	/* Aplicar la funcion de transferencia a la neurona de salida n */
	switch (tipo_fun_s) {
		/*La misma funcion de activacion que las neuronas de la capa oculta*/
		case IGUAL:{
			salida = activacion_neurona_oculta(net);
			break;
		}
		/*Lineal*/
		case LINEAL: default:{
			salida = net;
			break;
		}
	}

	//Resultado
	return salida;
}

/**
 * Evaluacion de la derivada con la salida obtenida por una neurona de salida
 *
 * Parametros:
 * 		double out		- Salida de la neurona de salida
 */
double derivada_neurona_salida(double out){
	//Variables
	double derivada;		//Valor de la derivada evaluada en la salida proporcionada

	switch (tipo_fun_s) {
		/*Derivada de la funcion utilizada por las neuronas ocultas*/
		case IGUAL:{
			derivada = derivada_neurona_oculta(out);
			break;
		}
		/*Lineal*/
		case LINEAL: default:{
			derivada = 1.0;
			break;
		}
	}

	//Resultado
	return derivada;
}

/**
 * Calculo del ECM para los patrones especificados en el vector de indices proporcionado
 *
 * Parametros:
 * 		int *vecIndx		- Referencia al arreglo con los indeces a considerar en el calculo
 * 		int cantIndx		- Cantidad de indices a considerar
 */
double calcular_ecm(int *vecIndx, int cantIndx){
	//Variables
	int i, k, m;				//Contadores
	double errTotal;			//ECM alcanzado
	double errorPatron;			//Error cuadratico por patron

	/* Calcular el error definitivo de los patrones proporcionados*/
	errTotal = 0.0;
	for (k = 0; k < cantIndx; k++){
		//Indice del patron
		i = vecIndx[k];

		//Calculo del error del patron k indexado por i
		errorPatron = 0.0;
		for (m = 0; m < nunit[nhlayer + 1]; m++)
			errorPatron += pow2(target[i][m] - outpt[i][m]) * 0.5;

		//Acumulacion del error cuadratico
		errTotal += errorPatron;
	}

	/* Error cuadratico medio promediado*/
	errTotal /= cantIndx;

	//Error resultante
	return errTotal;
}

/**
 * Calculo del factor de correlacion multiple entre los resultados especificados en el vector de indices dado
 *
 * Parametros:
 * 		int *vecIndx		- Referencia al arreglo con los indeces a considerar en el calculo
 * 		int cantIndx		- Cantidad de indices a considerar
 * 		double *corrM		- Referencia al arreglo que almacenara las correlaciones de cada rasgo de salida
 */
void calcular_correlacion(int *vecIndx, int cantIndx, double *corrM){
	int m, k, i;				//Contadores
	double *mediaT;				//Valores promedios de las salidas deseadas
	double *mediaO;				//Valores promedios de las salidas obtenidas
	double sum1, sum2, sum3;	//Variables auxiliares para el calculo de la correlacion

	//Reserva de espacio de memoria
	mediaT = (double *) calloc(nunit[nhlayer + 1], sizeof(double));
	mediaO = (double *) calloc(nunit[nhlayer + 1], sizeof(double));

	/* Inicializacion de valores promedios */
	for (m = 0; m < nunit[nhlayer + 1]; m++){
		mediaT[m] = 0.0;
		mediaO[m] = 0.0;
	}

	/* Acumulacion de suma para promedios*/
	for (k = 0; k < cantIndx; k++){
		i = vecIndx[k];
		for (m = 0; m < nunit[nhlayer + 1]; m++){
			mediaT[m] += target[i][m];
			mediaO[m] += outpt[i][m];
		}
	}

	//Calculo de valores promedios
	for (m = 0; m < nunit[nhlayer + 1]; m++){
		mediaT[m] /= cantIndx;
		mediaO[m] /= cantIndx;
	}

	/* Calculo de los valores de correlacion multiple de cada rasgo de salida */
	for (m = 0; m < nunit[nhlayer + 1]; m++){
		sum1 = 0.0; sum2 = 0.0; sum3 = 0.0;
		for (k = 0; k < cantIndx; k++){
			i = vecIndx[k];
			sum1 += (target[i][m] - mediaT[m]) * (outpt[i][m] - mediaO[m]);
			sum2 += pow2(target[i][m] - mediaT[m]);
			sum3 += pow2(outpt[i][m] - mediaO[m]);
		}
		//Correlacion multiple del rasgo de salida m
		corrM[m] = sum1 / (sqrt(sum2) * sqrt(sum3) + 0.00001);
	}

	//Liberacion de memoria
	free(mediaT);
	free(mediaO);
}

/* calculo bottom_up de la red para patron de entrada i */
void forward(int i) {
	//Variables
	int m, n, p;			//Contadores
	int offset;				//Desplazo para la correcta ubicacion en el vector con los pesos
	float net;				//Suma ponderada de valores a evaluar por la funcion de activacion de una neurona

	/* Calculo de salida para el nivel de entrada */
	for (m = 0; m < ninattr; m++){
		*(outptr[0] + m) = input[i][m];
	}

	/* Calculo de activacion para las neuronas en las capas escondidas y de salida */
	for (m = 1; m < (nhlayer + 2); m++) {
		//Recorrido de cada neurona de la capa m
		for (n = 0; n < nunit[m]; n++) {
			//Calculo de la senial de entrada para la neurona n
			net = 0.0;
			offset = (nunit[m - 1] + 1) * n;

			for (p = 0; p < (nunit[m - 1] + 1); p++){
				net += (*(wtptr[m - 1] + offset + p)) * (*(outptr[m - 1] + p));
			}

			//Neurona de la capa oculta
			if (m < (nhlayer + 1)){
				*(outptr[m] + n) = activacion_neurona_oculta(net);
			}
			//Neurona de la capa de salida
			else{
				*(outptr[m] + n) = activacion_neurona_salida(net);
			}
		}
	}

	for (n = 0; n < nunit[nhlayer + 1]; n++)
		outpt[i][n] = *(outptr[nhlayer + 1] + n);
} /* fin forward */

/* chequeo de diversas condiciones para verificar si el aprendizaje ha de
 terminar  */
int introspective() {
	/* se ha alcanzado maxima iteracion  */
	if (cnt >= cnt_num)
		return (MALEXIT);

	/* Continuar calculos */
	return (CONTCALC);
} /* fin introspective */

/* umbrales se tratan como pesos de una conectividad virtual a un nodo
 cuyo valor de salida es siempre menos uno  *//* OJO */
int rumelhart() {
	//Variables
	int m, n, p, k;		//Contadores
	int offset;			//Desplazo para la correcta ubicacion en el vector con los pesos
	float out;			//Variabla actual para almacenar temporalmente la salida de una neurona
	int i;				//Indice del patron actual
	time_t time_act;	//Tiempo de ejecucion actual

	//Inicializacion
	cnt = 0;
	result = CONTCALC;

	do { /* Iteraciones de la red (proceso de aprendizaje) */

		/* Por cada iteracion completa (epoch), calcular el error del sistema */
		errEntrenamientoNorm = 0.0;

		/* Permutacion del vector con los indices de los patrones de entrenamientos */
		/* con la intencion de presentar los patrones de forma aleatoria por cada epoch */
		permutar_vector(indxEntrenamiento, cantEntrenamiento);

		/* Para cada patron */
		for (k = 0; k < cantEntrenamiento; k++) { /* presentar a la red cada patron */
			/* Indice del patron a evaluar */
			i = indxEntrenamiento[k];

			/* Calculo bottom_up */
			forward(i);

			/* Propagacion del error top_down */

			/* Error del nivel de salida */
			for (m = 0; m < nunit[nhlayer + 1]; m++) {
				out = *(outptr[nhlayer + 1] + m);

				/* Calculo del gradiente local de las neuronas de salidas
				 * segun la funcion de transferencia utilizada*/
				*(errptr[nhlayer + 1] + m) = (target[i][m] - out) * derivada_neurona_salida(out);
			} /*de las UPs de la capa de salida, depende tipo funcion*/

			/* Calculo de los deltas para cada peso de la red */
			for (m = (nhlayer + 1); m >= 1; m--) { /* Moverse de capa salida a capa entrada */
				for (n = 0; n < (nunit[m - 1] + 1); n++) { /* Para cada unidad de la capa oculta/entrada */
					*(errptr[m - 1] + n) = 0.0; /* Inicializar su error en 0 */

					for (p = 0; p < nunit[m]; p++) { /* Para cada unidad de capa salida/oculta */
						/*Pesos ordenados segun UP salida/oculta*/
						offset = (nunit[m - 1] + 1) * p + n;

						/*Al ir variando por todos los patrones (indice i), */
						/*se logra la sumatoria del delta wij/wjk, considerando que */
						/*todos los terminos pueden entrar en la sumatoria*/

						/*Calculo del delta wij/wjk, con termino de momento*/
						*(delw[m - 1] + offset) = eta * (*(errptr[m] + p)) * (*(outptr[m - 1] + n))
						        + alfa * (*(delw[m - 1] + offset));

						/*Acumula*/
						*(errptr[m - 1] + n) += *(errptr[m] + p) * (*(wtptr[m - 1] + offset));
					} /*, termino sumatoria del delta wjk*/

					/*Calculo del gradiente local de las UPs de la capa oculta segun la funcion de transferencia*/
					*(errptr[m - 1] + n) = *(errptr[m - 1] + n) * derivada_neurona_oculta(*(outptr[m - 1] + n));

				}
			}

			/* Alteracion de pesos */
			for (m = 1; m < (nhlayer + 2); m++) {
				for (n = 0; n < nunit[m]; n++) { /*para cada unidad de la capa oculta/salida*/
					for (p = 0; p < (nunit[m - 1] + 1); p++) { /*para cada unidad de capa entrada/oculta*/
						offset = (nunit[m - 1] + 1) * n + p; /*ordenados segun UP oculta/salida*/
						*(wtptr[m - 1] + offset) += *(delw[m - 1] + offset);
					}
				}
			}

			/*Calcular el error para ese patron i*/
			ep[i] = 0.0;
			for (m = 0; m < nunit[nhlayer + 1]; m++) { /*moverse por unidades de capa salida*/
				ep[i] += pow2(target[i][m] - (*(outptr[nhlayer + 1] + m))) * 0.5;
			}
			errEntrenamientoNorm += ep[i];
		} /*presentacion de todos los patrones a la red*/

		/* Error normalizado del sistema */
		errEntrenamientoNorm /= cantEntrenamiento;

		//Se desean reportes y la iteracion actual es multiplo del valor proporcionado
		if ((iter_rep > 0) && (genMsj) && (((cnt + 1) % iter_rep) == 0 )){
			//Especificacion de la generacion
			fprintf(stdout, "\nReporte de la iteracion %ld de la sesion actual de entrenamiento\n", cnt + 1);

			//Impresion de la cantidad de segundos transcurridos desde el inicio del programa
			time(&time_act);
			fprintf(stdout, "\tTiempo transcurrido desde el inicio del programa: %.2lfseg\n", difftime(time_act, time_inicial));

			//Impresion de ECM actual
			fprintf(stdout, "\tError cuadratico medio normalizado actual del entrenamiento: %.5lf\n", errEntrenamientoNorm);

		}

		//Incremento del contador de iteraciones
		cnt++;

		/* Chequea condiciones para concluir aprendizaje */
		result = introspective();
	} while (result == CONTCALC);

	/* Actualiza salida con pesos alterados */
	for (k = 0; k < cantEntrenamiento; k++) /*moverse por todos los patrones*/
		forward(indxEntrenamiento[k]);

	//Tipo de culminacion del aprendizaje
	return (result);
} /* fin rumelhart */

/* Cuerpo principal de aprendizaje */
int entrenamiento(void) {
	int estado;					//Estado retornado por el entrenamiento

	//Inicializacion de valores de pesos
	initwt();

	//Proceso de entrenamiento mediante retropropagacion del error
	estado = rumelhart();

	/** Calculo del ECM normalizado del entrenamiento **/
	errEntrenamientoNorm = calcular_ecm(indxEntrenamiento, cantEntrenamiento);

	/** Calculo del factor de correlacion multiple del entrenamiento**/
	calcular_correlacion(indxEntrenamiento, cantEntrenamiento, corrMultEntr);

	//Fin
	return estado;
} /* fin learning */

/* Proceso de interrogatorio de la red con una arquitectura ya definida */
void interrogatorio(void){
	//Variables
	int k;				//Contador

	/* Genera una salida con los pesos acuales de la red */
	for (k = 0; k < cantInterrogatorio; k++) /*moverse por todos los patrones*/
		forward(indxInterrogatorio[k]);

	/* Calcular el error de prediccion*/
	errInterrogatorioNorm = calcular_ecm(indxInterrogatorio, cantInterrogatorio);

	/** Calculo del factor de correlacion multiple del interrogatorio**/
	calcular_correlacion(indxInterrogatorio, cantInterrogatorio, corrMultInt);
}

/**
 * Crear archivo para reportar los resultados globales obtenidos en las diversas corridas
 * El nombre del archivo estara compuesto por:
 * 	- Nombre del archivo con los patrones.
 * 	- Sufijo  _rg.red
 *
 **/
void reporte_global_corridas(){
	//Variables
	int c, m;					//Contadores
	char *var_file_name;		//Nombre del archivo
	FILE *frg;					//Archivo en donde se mostraran los resultados
	int tam;					//Tamanio auxiliar
	char aux[30];				//Auxiliar
	time_t time_act;			//Tiempo actual

	//Generacion del nombre del archivo
	tam = strlen(fnombre);
	if (rutaSalida != NULL)
		tam += strlen(rutaSalida);
	var_file_name = (char *) calloc(tam + 10, sizeof(char));
	strcpy(var_file_name, "");
	if (rutaSalida != NULL)
		strcpy(var_file_name, rutaSalida);
	strcat(var_file_name, fnombre);
	sprintf(var_file_name, "%s_g.red", var_file_name);

	//Validacion de la apertura del archivo
	if ((frg = fopen(var_file_name, "w")) == NULL) {
		mostrar_error("El archivo de reporte '");
		mostrar_error(var_file_name);
		mostrar_error("' no pudo crearse exitosamente\n");

		free(var_file_name);
		finalizar_programa(-1);
	}

	/* Encabezado */
	fprintf(frg, "UNIVERSIDAD DE CARABOBO\n");
	fprintf(frg, "FACULTAD EXPERIMENTAL DE CIENCIAS Y TECNOLOGIA\n");
	fprintf(frg, "RED NEURONAL MULTICAPAS UNIDIRECCIONAL\n");
	fprintf(frg, "APRENDIZAJE: RETROPROPAGACION DEL ERROR\n\n");
	fprintf(frg, "					      TRABAJO ESPECIAL DE GRADO\n\n");
	fprintf(frg, " \t\t'Algoritmo Genetico Paralelo para la Entonacion de Parametros de una\n");
	fprintf(frg, " \t\t       Red Neuronal Artificial de Retropropagación del Error'\n\n");
	fprintf(frg, "AUTOR: ORLANDO CASTILLO\n\n");
	fprintf(frg, "             REPORTE: RESULTADOS GLOBALES DE CORRIDAS");

	/* Parametros de arquitectura */
	fprintf(frg, "\n\n            PARAMETROS DE ARQUITECTURA FIJOS");
	fprintf(frg,"\n\nConstante de aprendizaje (eta)            = %.5f", eta);
	fprintf(frg,"\nRazon de momento (alfa)                   = %.3f", alfa);
	fprintf(frg,"\nNo maximo de iteraciones                  = %ld", cnt_num);
	fprintf(frg,"\nFuncion de transferencia (capas ocultas)  = ");
	switch (tipo_fun_o){
		case SIGMOIDE:
			fprintf(frg,"SIGMOIDE");
			strcpy(aux, "SIGMOIDE");
			break;
		case TANGENTE: default:
			fprintf(frg,"TANGENTE HIPERBOLICA");
			strcpy(aux, "TANGENTE HIPERBOLICA");
			break;
	}
	fprintf(frg,"\nFuncion de transferencia (capa de salida) = ");
	switch (tipo_fun_s){
		case IGUAL:
			fprintf(frg,"%s", aux);
			break;
		case TANGENTE: default:
			fprintf(frg,"LINEAL");
			break;
	}
	fprintf(frg,"\nNo de capas ocultas                       = %d", nhlayer);
	fprintf(frg,"\nNo de unidades por capa");
	fprintf(frg,"\n\tCapa de Entrada: %d", nunit[0]);
	fprintf(frg,"\n\tCapa Oculta 1  : %d", nunit[1]);
	if (nhlayer == 2)
		fprintf(frg,"\n\tCapa Oculta 2  : %d", nunit[2]);
	fprintf(frg,"\n\tCapa de Salida : %d", nunit[nhlayer + 1]);

	/* Datos generales del interrogatorio*/
	fprintf(frg, "\n\n                  ESTADISTICAS GLOBALES");
	fprintf(frg, "\n\nNo Patrones                                =  %-4d", ninput);
	fprintf(frg, "\nNro de rasgos de entrada                   =  %-4d", ninattr);
	fprintf(frg, "\nNro de rasgos de salida                    =  %-4d", noutattr);
	fprintf(frg, "\nNro Patrones usados en el entrenamiento    =  %-4d", cantEntrenamiento);
	fprintf(frg, "\nNro Patrones usados en el interrogatorio   =  %-4d", cantInterrogatorio);
	fprintf(frg, "\nNro de sesiones de Entr./Inte.             =  %-4d", cantRep);
	fprintf(frg, "\nECM promedio de entrenamientos normalizado =  %-14.8lf", ecmEntrPromNorm);
	fprintf(frg, "\nECM promedio de predicciones normalizado   =  %-14.8lf", ecmPredPromNorm);
	if (noutattr == 1){
		fprintf(frg, "\nCorrelacion promedio de entrenamientos     =  %-14.8lf", corrMultEntrProm[0]);
		fprintf(frg, "\nCorrelacion promedio de interrogatorios    =  %-14.8lf", corrMultIntProm[0]);
	}
	else{
		fprintf(frg, "\nCorrelaciones promedio de entrenamientos:\n");
		for (m = 0; m < nunit[nhlayer + 1]; m++){
			fprintf(frg, "\tCorrelacion del rasgo %d: %-14.8lf\n", m + 1, corrMultEntrProm[m]);
		}
		fprintf(frg, "\nCorrelaciones promedio de interrogatorios:\n");
		for (m = 0; m < nunit[nhlayer + 1]; m++){
			fprintf(frg, "\tCorrelacion del rasgo %d: %-14.8lf\n", m + 1, corrMultIntProm[m]);
		}
	}
	time(&time_act);
	fprintf(frg, "\nTiempo estimado de ejecucion               =  %.2lfseg\n", difftime(time_act, time_inicial));

	//Informacion de los reportes generados
	fprintf(frg, "\n\n                   REPORTES GENERADOS");
	fprintf(frg, "\n\nEl programa genera el reporte actual que contiene informacion general. Adicionalmente,\n");
	fprintf(frg, "por cada sesion de entrenamiento/interrogatorio se generan dos reportes. Como se\n");
	if (cantRep > 1){
		fprintf(frg, "solicitaron realizar %d sesiones, se generaron un total de %d reportes. Los archivos que se generan por cada\n",
				cantRep, 2 *cantRep);
	}
	else{
		fprintf(frg, "solicito realizar 1 sesion, se generaron un total de 2 reportes. Los archivos que se generan por cada\n");
	}
	fprintf(frg, "sesion son los siguientes (Los asteriscos indican numeros enteros en el intervalo [1, %d]).:\n\n", cantRep);
	fprintf(frg, "\t - %s_r*.red: Resultados del entrenamiento e interrogatorio * realizado a la mejor red obtenida.\n", fnombre);
	fprintf(frg, "\t - %s_a*.red: Arquitectura de la mejor red generada durante la sesion de entrenamiento *.\n\n", fnombre);

	/* Clausura de archivo */
	if ((c = fclose(frg)) != 0) {
		mostrar_error("El archivo de reporte '");
		mostrar_error(var_file_name);
		mostrar_error("' no pudo cerrarse exitosamente\n");

		free(var_file_name);
		finalizar_programa(-1);
	}
	free(var_file_name);

}

/**
 * Crear archivo para reportar los resultados de la sesion de entrenamiento/interrogatorio especificada por parametro. Si el valor
 * de la corrida es 0, se asume entonces que solo se realiza el entrenamiento de la red
 * El nombre del archivo estara compuesto por:
 * 	- Nombre del archivo con los patrones.
 * 	- Sufijo  _r%c%.red, donde %c% corresponde a el valor de la corrida que se mostrara, si corrida es mayor que 0
 * 			  _e.red, si corrida es igual a 0
 *
 * Parametro:
 * 		int corrida			- Identificador de la sesion realizada
 *
 **/
void reporte_entrenamiento_interrogatorio(int corrida){
	//Variables
	int c, k, m, ii, jj;		//Contadores
	double err_c;				//Error cuadratico del patron en evaluacion
	double err;					//Error del patron en evaluacion
	double errPorc;				//Error porcentual del patron en evaluacion
	char *var_file_name;		//Nombre del archivo
	FILE *fri;					//Archivo en donde se mostraran los resultados
	int tam;					//Tamanio auxiliar
	int *refPat;				//Referencia al vector de indices a mostrar
	int cantPat;				//Cantidad de patrones en el grupo
	double errPred;				//ECM de predicion
	double errEnt;				//ECM del entrenamiento
	int contBuenos = 0;			//Cantidad de valores con buena prediccion
	int contMedios = 0;			//Cantidad de valores con prediccion regular
	int contMalos = 0;			//Cantidad de valores con mala prediccion
	int iter;					//Auxiliar para la iteracion de impresion de errores

	//Calculo de los ECM de entrenamiento
	errEnt = calcular_ecm(indxEntrenamiento, cantEntrenamiento);
	qsort(indxEntrenamiento, cantEntrenamiento, sizeof(int), cmp_int);

	//Calculo de los ECM de prediccion
	if (corrida > 0){
		errPred = calcular_ecm(indxInterrogatorio, cantInterrogatorio);
		qsort(indxInterrogatorio, cantInterrogatorio, sizeof(int), cmp_int);
	}

	//Generacion del nombre del archivo
	tam = strlen(fnombre);
	if (rutaSalida != NULL)
		tam += strlen(rutaSalida);
	var_file_name = (char *) calloc(tam + 10, sizeof(char));
	strcpy(var_file_name, "");
	if (rutaSalida != NULL)
		strcpy(var_file_name, rutaSalida);
	strcat(var_file_name, fnombre);
	//Sufijo correcto
	if (corrida > 0)
		sprintf(var_file_name, "%s_r%d.red", var_file_name, corrida);
	else
		sprintf(var_file_name, "%s_e.red", var_file_name);

	//Validacion de la apertura del archivo
	if ((fri = fopen(var_file_name, "w")) == NULL) {
		mostrar_error("El archivo de reporte '");
		mostrar_error(var_file_name);
		mostrar_error("' no pudo crearse exitosamente\n");

		free(var_file_name);
		finalizar_programa(-1);
	}

	/* Encabezado */
	fprintf(fri, "UNIVERSIDAD DE CARABOBO\n");
	fprintf(fri, "FACULTAD EXPERIMENTAL DE CIENCIAS Y TECNOLOGIA\n");
	fprintf(fri, "RED NEURONAL MULTICAPAS UNIDIRECCIONAL\n");
	fprintf(fri, "APRENDIZAJE: RETROPROPAGACION DEL ERROR\n\n");
	fprintf(fri, "					      TRABAJO ESPECIAL DE GRADO\n\n");
	fprintf(fri, " \t\t'Algoritmo Genetico Paralelo para la Entonacion de Parametros de una\n");
	fprintf(fri, " \t\t       Red Neuronal Artificial de Retropropagación del Error'\n\n");
	fprintf(fri, "AUTOR: ORLANDO CASTILLO\n\n");

	/* Datos generales */
	if (corrida > 0){
		fprintf(fri, "             REPORTE: RESULTADOS DEL ENTRENAMIENTO E INTERROGATORIO");
		fprintf(fri, "\n                               CORRIDA: %d", corrida);
	}
	else
		fprintf(fri, "                       REPORTE: RESULTADOS DEL ENTRENAMIENTO");

	fprintf(fri, "\n\nReporte de arquitectura          =  %s_a%d.red", fnombre, corrida);
	fprintf(fri, "\nNo Patrones                      =  %-4d", ninput);
	fprintf(fri, "\nNro de rasgos de entrada         =  %-4d", ninattr);
	fprintf(fri, "\nNro de rasgos de salida          =  %-4d", noutattr);
	fprintf(fri, "\nECM del entrenamiento            =  %-14.8lf", errEnt);
	if (corrida > 0)
		fprintf(fri, "\nECM de prediccion                =  %-14.8lf", errPred);
	if (noutattr == 1){
		fprintf(fri, "\nCorrelacion de entrenamiento     =  %-14.8lf", corrMultEntr[0]);
		if (corrida > 0)
			fprintf(fri, "\nCorrelacion de interrogatorio    =  %-14.8lf", corrMultInt[0]);
	}
	else{
		fprintf(fri, "\nCorrelaciones de entrenamiento:\n");
		for (m = 0; m < nunit[nhlayer + 1]; m++){
			fprintf(fri, "\tCorrelacion del rasgo %d: %-14.8lf\n", m + 1, corrMultEntr[m]);
		}
		if (corrida > 0){
			fprintf(fri, "\nCorrelaciones de interrogatorio:\n");
			for (m = 0; m < nunit[nhlayer + 1]; m++){
				fprintf(fri, "\tCorrelacion del rasgo %d: %-14.8lf\n", m + 1, corrMultInt[m]);
			}
		}
	}

	/* Impresion de errores por patrones, clasificados por grupos de patrones de entrenamiento y de interrogatorio */
	if (corrida > 0)
		iter = 2;
	else
		iter = 1;
	for (ii = 0; ii < iter; ii++){
		//Impresion de datos de entrenamiento
		if (ii == 0){
			fprintf(fri, "\n\n                  ERRORES DE LA FASE DE ENTRENAMIENTO");
			fprintf(fri, "\n\n%% Patrones de Entrenamiento      =  %d%%", PORC_ENTR);
			fprintf(fri, "\nNo Patrones de Entrenamiento     =  %-4d", cantEntrenamiento);
			refPat = indxEntrenamiento;
			cantPat = cantEntrenamiento;
		}
		//Impresion de datos de entrenamiento
		else{
			fprintf(fri, "\n\n                  ERRORES DE LA FASE DE INTERROGATORIO");
			fprintf(fri, "\n\n%% Patrones de Interrogatorio     =  %d%%", 100 - PORC_ENTR);
			fprintf(fri, "\nNo Patrones de Interrogatorio    =  %-4d", cantInterrogatorio);
			refPat = indxInterrogatorio;
			cantPat = cantInterrogatorio;
			contBuenos = 0;
			contMedios = 0;
			contMalos = 0;
		}

		/* Solo para patrones de salida con 1 componente */
		fprintf(fri, "\n\nListado de los errores obtenidos patron a patron:\n\n");
		if (noutattr == 1){
			PRINT_LINE(k, 99, fri);
			fprintf(fri, "%-6s - %-15s - %-15s - %-16s - %-16s - %-16s\n", "Patron", "Salida Deseada", "Salida Generada",
					"Error cuadratico", "Error absoluto", "Error porcentual");
			PRINT_LINE(k, 99, fri);

			for (jj = 0; jj < cantPat; jj++) {
				k = refPat[jj];
				err = fabs(target[k][0] - outpt[k][0]);
				err_c = pow2(err) * 0.5;
				errPorc = (err * 100.0) / fabs(target[k][0] + 0.00001);
				fprintf(fri, "%-6d   %-15.5f   %-15.5f   %-16.5lf   %-16.5lf   %-16.5lf\n", k + 1, target[k][0], outpt[k][0], err_c,
						err, errPorc);

				if (errPorc < 11)
					contBuenos++;
				else if (errPorc >= 11 && errPorc < 16)
					contMedios++;
				else
					contMalos++;
			}
			PRINT_LINE(k, 99, fri);
			fprintf(fri,"\nNumero de observaciones con %% Error < 11: %3d, representa: %6.2f %%",  contBuenos,
					(contBuenos * 100.0) / cantPat );
			fprintf(fri,"\nNumero de observaciones con 11 <= %% Error < 16: %3d, representa: %6.2f %%",  contMedios,
							(contMedios * 100.0) / cantPat );
			fprintf(fri,"\nNumero de observaciones con %% Error >= 16: %3d, representa: %6.2f %%", contMalos,
					(contMalos * 100.0) / cantPat);
		}
		/* Para patrones de salida con mas de 1 componente */
		else{
			PRINT_LINE(k, 25, fri);
			fprintf(fri, "%-6s - %-16s\n", "Patron", "Error cuadratico");
			PRINT_LINE(k, 25, fri);

			for (jj = 0; jj < cantPat; jj++) {
				k = refPat[jj];
				err_c = 0.0;
				for (m = 0; m < nunit[nhlayer + 1]; m++)
					err_c += pow2(target[k][m] - outpt[k][m]) * 0.5;
				fprintf(fri, "%-6d   %-11.5f\n", k + 1, err_c);
			}
			PRINT_LINE(k, 25, fri);
		}
	}

	if ((c = fclose(fri)) != 0) {
		mostrar_error("El archivo de reporte '");
		mostrar_error(var_file_name);
		mostrar_error("' no pudo cerrarse exitosamente\n");

		free(var_file_name);
		finalizar_programa(-1);
	}
	free(var_file_name);
}

/**
 * Se carga una red a partir de un archivo con una arquitectura ya definida
 */
void cargar_red(){
	//Variables
	int i, j, k;			//Contadores
	char *file;				//Ruta y nombre del archivo con los patrones
	FILE *farquitectura;	//Archivo con los patrones de entrada y salida
	BOOL valido = TRUE;		//Determina la validez del formato del archivo

	//Combinacion de la ruta y nombre del archivo con la arquitectura
	file = (char *) calloc(strlen(fruta_a) + strlen(fnombre_a) + 10, sizeof(char));
	strcat(file, fruta_a);
	strcat(file, fnombre_a);
	strcat(file, ".red");

	//Si la ruta resulta invalida, se reporta el error
	if ((farquitectura = fopen(file, "r")) == NULL){
		mostrar_error("No fue posible abrir el archivo de la arquitectura '");
		mostrar_error(fnombre);
		mostrar_error("' exitosamente. Verifique que la ruta proporcionada al directorio '");
		mostrar_error(fruta);
		mostrar_error("' sea valida y que el archivo sea de tipo '.red'\n");
		free(file);
		finalizar_programa(-1);
	}
	free(file);

	//Lectura de la constante de aprendizaje y razon de momentum
	if (fscanf(farquitectura, "%f %f", &eta, &alfa) == EOF)
		valido = FALSE;

	//Lectura de la cantidad maxima de iteraciones
	if (fscanf(farquitectura, "%ld", &cnt_num) == EOF)
		valido = FALSE;

	//Funciones de activaciones de la capa oculta y capa de salida
	if (fscanf(farquitectura, "%d %d", &tipo_fun_o, &tipo_fun_s) == EOF)
		valido = FALSE;

	//Lectura de las capas ocultas
	if (fscanf(farquitectura, "%d", &nhlayer) == EOF)
		valido = FALSE;
	if (nhlayer == 2){
		if (fscanf(farquitectura,"%d %d", &(nunit[1]), &(nunit[2])) == EOF)
			valido = FALSE;
	}
	else{
		if (fscanf(farquitectura,"%d", &(nunit[1])) == EOF)
			valido = FALSE;
	}

	//Fomato valido hasta ahora
	if (valido){
		//Asignacion de recursos a estructuras de la red
		init_red();

		//Lectura de los pesos de la red
		for (i = 0; i < (nhlayer + 1) && valido; i++) {
			for (j = 0; j < nunit[i + 1] && valido; j++){
				for (k = 0; k < (nunit[i] + 1) && valido; k++){
					if (fscanf(farquitectura, "%f", (wtptr[i] + (j * (nunit[i] + 1)) + k)) == EOF)
						valido = FALSE;
				}
			}
		}
	}

	//Cierre de archivo
	fclose(farquitectura);

	//Formato de archivo invalido
	if (!valido){
		mostrar_error("El archivo de arquitectura '");
		mostrar_error(fnombre);
		mostrar_error("' no cumple con el formato requerido, revise el manual de usuario distribuido\n");
		finalizar_programa(-1);
	}
}

/**
 * Crear archivo para reportar los resultados del interrogatorio a todos los patrones con la arquitectura actual
 * El nombre del archivo estara compuesto por:
 * 	- Nombre del archivo con los patrones.
 * 	- Sufijo  _i.red
 **/
void reporte_interrogatorio(){
	//Variables
	int c, k, m, ii, jj;		//Contadores
	double err;					//Error del patron en evaluacion
	double errPorc;				//Error porcentual del patron en evaluacion
	char *var_file_name;		//Nombre del archivo
	double errTotal;			//Error del sistema relativo a los patrones a mostrar
	FILE *fri;					//Archivo en donde se mostraran los resultados
	int tam;					//Tamanio auxiliar
	double *mediaT;				//Valores promedios de las salidas deseadas
	double *mediaO;				//Valores promedios de las salidas obtenidas
	double *vecECM;				//Errores cuadraticos de cada patron
	double *corrM;				//Valores de correlacion multiple por cada rasgo de salida
	double sum1, sum2, sum3;	//Variables auxiliares para el calculo de la correlacion
	int *refPat;				//Referencia al vector de indices a mostrar
	int cantPat;				//Cantidad de patrones en el grupo
	double errPred;				//ECM de predicion
	int contBuenos = 0;			//Cantidad de valores con buena prediccion
	int contMedios = 0;			//Cantidad de valores con prediccion regular
	int contMalos = 0;			//Cantidad de valores con mala prediccion
	char aux[30];				//Auxiliar

	//Reserva de espacio de memoria
	mediaT = (double *) calloc(nunit[nhlayer + 1], sizeof(double));
	mediaO = (double *) calloc(nunit[nhlayer + 1], sizeof(double));
	corrM = (double *) calloc(nunit[nhlayer + 1], sizeof(double));
	vecECM = (double *) calloc(ninput, sizeof(double));

	/* Inicializacion de valores promedios */
	for (m = 0; m < nunit[nhlayer + 1]; m++){
		mediaT[m] = 0.0;
		mediaO[m] = 0.0;
	}

	/* Calcular el error definitivo del sistema por cada patron*/
	errTotal = 0.0;
	for (k = 0; k < ninput; k++){
		//Calculo del error del patron k y acumulacion de suma para promedios
		vecECM[k] = 0.0;
		for (m = 0; m < nunit[nhlayer + 1]; m++){
			vecECM[k] += pow2(target[k][m] - outpt[k][m]) * 0.5;
			mediaT[m] += target[k][m];
			mediaO[m] += outpt[k][m];
		}
		errTotal += vecECM[k];
	}
	errTotal /= ninput;

	//Calculo de valores promedios
	for (m = 0; m < nunit[nhlayer + 1]; m++){
		mediaT[m] /= ninput;
		mediaO[m] /= ninput;
	}

	/* Calculo de los valores de correlacion multiple de cada rasgo de salida */
	for (m = 0; m < nunit[nhlayer + 1]; m++){
		sum1 = 0.0; sum2 = 0.0; sum3 = 0.0;
		for (k = 0; k < ninput; k++){
			sum1 += (target[k][m] - mediaT[m]) * (outpt[k][m] - mediaO[m]);
			sum2 += pow2(target[k][m] - mediaT[m]);
			sum3 += pow2(outpt[k][m] - mediaO[m]);
		}
		//Correlacion multiple del rasgo de salida m
		corrM[m] = sum1 / (sqrt(sum2) * sqrt(sum3) + 0.00001);
	}

	/*Calculo del ECM de prediccion */
	errPred = 0.0;
	for (ii = 0; ii < cantInterrogatorio; ii++)
		errPred += vecECM[indxInterrogatorio[ii]];
	errPred /= cantInterrogatorio;

	//Ordenamiento de los indices
	qsort(indxInterrogatorio, cantInterrogatorio, sizeof(int), cmp_int);

	//Liberacion de espacio de memoria no necesario
	free(mediaT);
	free(mediaO);

	//Generacion del nombre del archivo
	tam = strlen(fnombre);
	if (rutaSalida != NULL)
		tam += strlen(rutaSalida);
	var_file_name = (char *) calloc(tam + 10, sizeof(char));
	strcpy(var_file_name, "");
	if (rutaSalida != NULL)
		strcpy(var_file_name, rutaSalida);
	strcat(var_file_name, fnombre);
	sprintf(var_file_name, "%s_i.red", var_file_name);

	//Validacion de la apertura del archivo
	if ((fri = fopen(var_file_name, "w")) == NULL) {
		mostrar_error("El archivo de reporte '");
		mostrar_error(var_file_name);
		mostrar_error("' no pudo crearse exitosamente\n");

		free(var_file_name);
		free(corrM);
		free(vecECM);
		finalizar_programa(-1);
	}

	/* Encabezado */
	fprintf(fri, "UNIVERSIDAD DE CARABOBO\n");
	fprintf(fri, "FACULTAD EXPERIMENTAL DE CIENCIAS Y TECNOLOGIA\n");
	fprintf(fri, "RED NEURONAL MULTICAPAS UNIDIRECCIONAL\n");
	fprintf(fri, "APRENDIZAJE: RETROPROPAGACION DEL ERROR\n\n");
	fprintf(fri, "					      TRABAJO ESPECIAL DE GRADO\n\n");
	fprintf(fri, " \t\t'Algoritmo Genetico Paralelo para la Entonacion de Parametros de una\n");
	fprintf(fri, " \t\t       Red Neuronal Artificial de Retropropagación del Error'\n\n");
	fprintf(fri, "AUTOR: ORLANDO CASTILLO\n\n");
	fprintf(fri, "             REPORTE: RESULTADOS DEL INTERROGATORIO");

	/* Parametros de arquitectura */
	fprintf(fri, "\n\n            PARAMETROS DE LA ARQUITECTURA ENTRENADA");
	fprintf(fri,"\n\nConstante de aprendizaje (eta)            = %.5f", eta);
	fprintf(fri,"\nRazon de momento (alfa)                   = %.3f", alfa);
	fprintf(fri,"\nNo maximo de iteraciones                  = %ld", cnt_num);
	fprintf(fri,"\nFuncion de transferencia (capas ocultas)  = ");
	switch (tipo_fun_o){
		case SIGMOIDE:
			fprintf(fri,"SIGMOIDE");
			strcpy(aux, "SIGMOIDE");
			break;
		case TANGENTE: default:
			fprintf(fri,"TANGENTE HIPERBOLICA");
			strcpy(aux, "TANGENTE HIPERBOLICA");
			break;
	}
	fprintf(fri,"\nFuncion de transferencia (capa de salida) = ");
	switch (tipo_fun_s){
		case IGUAL:
			fprintf(fri,"%s", aux);
			break;
		case TANGENTE: default:
			fprintf(fri,"LINEAL");
			break;
	}
	fprintf(fri,"\nNo de capas ocultas                       = %d", nhlayer);
	fprintf(fri,"\nNo de unidades por capa");
	fprintf(fri,"\n\tCapa de Entrada: %d", nunit[0]);
	fprintf(fri,"\n\tCapa Oculta 1  : %d", nunit[1]);
	if (nhlayer == 2)
		fprintf(fri,"\n\tCapa Oculta 2  : %d", nunit[2]);
	fprintf(fri,"\n\tCapa de Salida : %d", nunit[nhlayer + 1]);

	/* Datos generales del interrogatorio*/
	fprintf(fri, "\n\n                  ERRORES DEL INTERROGATORIO");
	fprintf(fri, "\n\nNo Patrones                      =  %-4d", ninput);
	fprintf(fri, "\nNro de rasgos de entrada         =  %-4d", ninattr);
	fprintf(fri, "\nNro de rasgos de salida          =  %-4d", noutattr);
	fprintf(fri, "\nECM de prediccion                =  %-14.8lf", errPred);
	if (noutattr == 1)
		fprintf(fri, "\nCorrelacion multiple             =  %-14.8lf", corrM[0]);
	else{
		fprintf(fri, "\nCorrelacion multiple de cada rasgo de salida:\n");
		for (m = 0; m < nunit[nhlayer + 1]; m++){
			fprintf(fri, "\tCorrelacion del rasgo %d: %-14.8lf\n", m + 1, corrM[m]);
		}
	}

	/* Impresion de errores por patrones */
	refPat = indxInterrogatorio;
	cantPat = cantInterrogatorio;
	contBuenos = 0;
	contMedios = 0;
	contMalos = 0;

	/* Solo para patrones de salida con 1 componente */
	fprintf(fri, "\n\nListado de los errores obtenidos patron a patron:\n\n");
	if (noutattr == 1){
		PRINT_LINE(k, 99, fri);
		fprintf(fri, "%-6s - %-15s - %-15s - %-16s - %-16s - %-16s\n", "Patron", "Salida Deseada", "Salida Generada",
				"Error cuadratico", "Error absoluto", "Error porcentual");
		PRINT_LINE(k, 99, fri);

		for (jj = 0; jj < cantPat; jj++) {
			k = refPat[jj];
			err = fabs(target[k][0] - outpt[k][0]);
			errPorc = (err * 100.0) / fabs(target[k][0] + 0.00001);
			fprintf(fri, "%-6d   %-15.5f   %-15.5f   %-16.5lf   %-16.5lf   %-16.5lf\n", k + 1, target[k][0], outpt[k][0], vecECM[k],
					err, errPorc);

			if (errPorc < 11)
				contBuenos++;
			else if (errPorc >= 11 && errPorc < 16)
				contMedios++;
			else
				contMalos++;
		}
		PRINT_LINE(k, 99, fri);
		fprintf(fri,"\nNumero de observaciones con %% Error < 11: %3d, representa: %6.2f %%",  contBuenos,
				(contBuenos * 100.0) / cantPat );
		fprintf(fri,"\nNumero de observaciones con 11 <= %% Error < 16: %3d, representa: %6.2f %%",  contMedios,
						(contMedios * 100.0) / cantPat );
		fprintf(fri,"\nNumero de observaciones con %% Error >= 16: %3d, representa: %6.2f %%", contMalos,
				(contMalos * 100.0) / cantPat);
	}
	/* Para patrones de salida con mas de 1 componente */
	else{
		PRINT_LINE(k, 25, fri);
		fprintf(fri, "%-6s - %-16s\n", "Patron", "Error cuadratico");
		PRINT_LINE(k, 25, fri);

		for (jj = 0; jj < cantPat; jj++) {
			k = refPat[jj];
			fprintf(fri, "%-6d   %-11.5f\n", k + 1, vecECM[k]);
		}
		PRINT_LINE(k, 25, fri);
	}

	//Liberacion de memoria
	free(corrM);
	free(vecECM);

	if ((c = fclose(fri)) != 0) {
		mostrar_error("El archivo de reporte '");
		mostrar_error(var_file_name);
		mostrar_error("' no pudo cerrarse exitosamente\n");

		free(var_file_name);
		finalizar_programa(-1);
	}
	free(var_file_name);
}

/**
 * Crear archivo para reportar los datos de la arquitectura de la red, incluyendo los pesos.
 * El nombre del archivo estara compuesto por:
 * 	- Nombre del archivo con los patrones.
 * 	- Sufijo  _a%c%.red, donde %c% corresponde a el valor de la corrida que se mostrara
 **/
void reporte_arquitectura(int corrida){
	//Variables
	int c, k, i, j;			//Contadores
	char *var_file_name;	//Nombre del archivo
	FILE *fa;				//Archivo en donde se mostraran la arquitectura
	int tam;				//Tamanio auxiliar

	//Generacion del nombre del archivo
	tam = strlen(fnombre);
	if (rutaSalida != NULL)
		tam += strlen(rutaSalida);
	var_file_name = (char *) calloc(tam + 30, sizeof(char));
	strcpy(var_file_name, "");
	if (rutaSalida != NULL)
		strcpy(var_file_name, rutaSalida);
	strcat(var_file_name, fnombre);

	//Sufijo correcto
	if (corrida > 0)
		sprintf(var_file_name, "%s_a%d.red", var_file_name, corrida);
	else
		sprintf(var_file_name, "%s_ae.red", var_file_name);

	//Validacion de la apertura del archivo
	if ((fa = fopen(var_file_name, "w")) == NULL) {
		mostrar_error("El archivo de reporte '");
		mostrar_error(var_file_name);
		mostrar_error("' no pudo crearse exitosamente\n");

		free(var_file_name);
		finalizar_programa(-1);
	}

	/* Parametros de la red */

	//Constante de aprendizaje y razon de momentum
	fprintf(fa, "%.5f %.5f\n", eta, alfa);

	//Cantidad maxima de iteraciones
	fprintf(fa, "%ld\n", cnt_num);

	//Funciones de activaciones de la capa oculta y capa de salida
	fprintf(fa, "%d %d\n", tipo_fun_o, tipo_fun_s);

	//Capas ocultas
	fprintf(fa, "%d ", nhlayer);
	if (nhlayer == 2)
		fprintf(fa,"%d %d\n", nunit[1], nunit[2]);
	else
		fprintf(fa,"%d\n", nunit[1]);

	//Pesos de la red
	for (i = 0; i < (nhlayer + 1); i++) {
		for (j = 0; j < nunit[i + 1]; j++){
			for (k = 0; k < (nunit[i] + 1); k++){
				fprintf(fa, "%.3f ", *(wtptr[i] + (j * (nunit[i] + 1)) + k));
			}
			fputc('\n', fa);
		}
	}

	if ((c = fclose(fa)) != 0) {
		mostrar_error("El archivo de reporte '");
		mostrar_error(var_file_name);
		mostrar_error("' no pudo cerrarse exitosamente\n");

		free(var_file_name);
		finalizar_programa(-1);
	}
	free(var_file_name);
}

//Prepara los recursos necesarios para la ejecucion del programa con el esquema seleccionado
void preparar_esquema_ejecucion(){
	//No se usara ningun esquema
	if (genReportes)
		return ;

	//Multiples formas de generar resultados
	switch (esquemaEjecucion){
		//Esquema con fork
		case FORK:{
			//Preparacion de los recursos del esquema que usa fork
			if (!preparar_fork_esclavo(idSegmento)){
				mostrar_error("No pudieron prepararse los recursos necesarios para el esquema de ejecucion con fork\n");
				finalizar_programa(-1);
			}
			break;
		}
		//Esquema con fork
		case MPI:{
			//Preparacion de los recursos del esquema que usa mpi
			if (!preparar_mpi_esclavo()){
				mostrar_error("No pudieron prepararse los recursos necesarios para el esquema de ejecucion con mpi\n");
				finalizar_programa(-1);
			}
			break;
		}
		//Esquema secuencial
		case SECUENCIAL: default:{
			break;		//No hace falta preparar nada
		}
	}
	//Se registra la inicializacion
	esquemaIniciado = TRUE;
}

//Libera los recursos necesarios para la ejecucion del programa con el esquema seleccionado
void liberar_esquema_ejecucion(){
	//No se usara ningun esquema
	if (genReportes)
		return ;

	//Se registra la liberacion
	esquemaIniciado = FALSE;

	//Multiples posibilidades para la liberacion
	switch (esquemaEjecucion){
		//Esquema con fork
		case FORK:{
			//Liberacion de los recursos del esquema que usa fork
			if (!liberar_fork()){
				mostrar_error("No pudieron liberarse los recursos creados para el esquema con fork\n");
				finalizar_programa(-1);
			}
			break;
		}
		//Esquema con mpi
		case MPI:{
			//Liberacion de los recursos del esquema que usa mpi
			if (!liberar_mpi()){
				mostrar_error("No pudieron liberarse los recursos creados para el esquema con mpi\n");
				finalizar_programa(-1);
			}
			break;
		}
		//Esquema secuencial
		case SECUENCIAL: default:{
			break;			//No hace falta liberar nada
		}
	}
}

/**
 * Se generan los resultados de acuerdo al esquema de ejecucion utilizado
 */
void generar_resultados(){
	//No se usara ningun esquema
	if (genReportes)
		return ;

	//Multiples formas de generar resultados
	switch (esquemaEjecucion){
		//Esquema con fork
		case FORK:{
			//Se escriben los resultados en la memoria compartida,
			if (!enviar_error_fork(posMem, ecmPredPromNorm)){
				mostrar_error("No pudo enviarse correctamente el error bajo el esquema de ejecucion fork\n");
				finalizar_programa(-1);
			}
			break;
		}
		//Esquema con fork
		case MPI:{
			//Se envian los resultados a traves del paso de mensajes
			if (!enviar_error_mpi(posMem, ecmPredPromNorm)){
				mostrar_error("No pudo enviarse correctamente el error bajo el esquema de ejecucion mpi\n");
				finalizar_programa(-1);
			}
			break;
		}
		//Esquema secuencial
		case SECUENCIAL: default:{
			//Se envia el ECM promedio por medio de archivos
			if (!enviar_error_secuencial(ecmPredPromNorm)){
				mostrar_error("No pudo enviarse correctamente el error bajo el esquema de ejecucion secuencial\n");
				finalizar_programa(-1);
			}
			break;
		}
	}
}

//Liberacion de los recursos dinamicos asociados por el programa
void liberar_recursos(){
	//Liberacion de la memoria asignada a los patrones
	if (patronesIniciados)
		liberar_patrones();

	//Liberacion de la memoria asignada a los parametros
	if (parametrosIniciados)
		liberar_parametros();

	//Liberacion de memoria dinamica para estructuras de la red
	if (redIniciada)
		liberar_red();

	//Liberacion del esquema de ejecucion
	if (esquemaIniciado)
		liberar_esquema_ejecucion();

	//Liberacion de recursos asignados para la division de patrones
	if (patronesDiv)
		liberar_division();
}

//Permite terminar la ejecucion del programa, considerando la limpieza de los recursos dinamicos utilizados
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

/**************************************************************/
/***************   CUERPO PRINCIPAL   *************************/
/**************************************************************/

int main(int argc, char *argv[]) {
	//Variables
	char *msj;					//Almacena un mensaje en caso de existir
	int ii, jj;					//Contadores

	//Iniciar el reloj de ejecucion
	time(&time_inicial);

	//Verificacion de la especificacion correcta de parametros
	if ((msj = parametros_entrada(argc, argv)) != NULL){
		mostrar_error(msj);
		free(msj);
		finalizar_programa(-1);
	}

	//Preparacion de los recursos asociados al esquema de ejecucion seleccionado
	preparar_esquema_ejecucion();

	//Se se desean reportes, entonces se imprime un mensaje de inicio exitoso
	if (genMsj){
		fprintf(stdout, "\nHa iniciado la ejecucion de la red neuronal de retropropagacion del error...\n");
	}

	//Semilla aleatoria
	srand(time(NULL));

	//Lectura de los patrones de entrada
	init_patrones();

	//Modo de sesiones de entrenamiento e interrogatorio
	if (!modoInterrogatorio){
		//Asignacion de memoria dinamica para estructuras de la red
		init_red();

		//Normalizacion de los patrones
		normalizar();

		//Modo para solo entrenamiento
		if (cantRep == 0){
			//Se asignan todos los patrones al grupo de entrenamiento
			definir_division(100);

			//Asignar patrones
			asignar_patrones();

			//Fase de entrenamiento
			if (genMsj)
				fprintf(stdout, "\nIniciando la sesion de entrenamiento...\n");
			entrenamiento();
			if (genMsj){
				fprintf(stdout, "\nCulminada la sesion de entrenamiento\n");
				fprintf(stdout, "\tError cuadratico normalizado de entrenamiento alcanzado: %.5lf\n", errEntrenamientoNorm);
			}

			//Desnormalizacion de los patrones
			desnormalizar();

			//Generacion de un reporte los resultados de la actual sesion de entrenamiento/interrogatorio
			reporte_entrenamiento_interrogatorio(0);

			//Generacion de un reporte con los datos de la arquitectura de la red
			reporte_arquitectura(0);
		}
		else{
			//Division de patrones
			definir_division(PORC_ENTR);

			//Inicializacion
			ecmPredPromNorm = 0.0;
			ecmEntrPromNorm = 0.0;
			for (ii = 0; ii < noutattr; ii++){
				corrMultEntrProm[ii] = 0.0;
				corrMultIntProm[ii] = 0.0;
			}

			//Proceso de repeticion de sesiones de entrenamiento e interrogatorio
			for (ii = 0; ii < cantRep; ii++){
				//Asignar patrones de forma aleatoria
				asignar_patrones();

				//Fase de entrenamiento
				if (genMsj)
					fprintf(stdout, "\nIniciando la sesion de entrenamiento %d...\n", ii + 1);
				entrenamiento();
				if (genMsj){
					fprintf(stdout, "\nCulminada la sesion de entrenamiento %d\n", ii + 1);
					fprintf(stdout, "\tError cuadratico normalizado de entrenamiento alcanzado: %.5lf\n", errEntrenamientoNorm);
				}

				//Acumulacion de la suma de ECM de los entrenamiento
				ecmEntrPromNorm += errEntrenamientoNorm;

				//Acumulacion de las correlaciones del entrenamiento
				for (jj = 0; jj < noutattr; jj++)
					corrMultEntrProm[jj] += corrMultEntr[jj];

				//Fase de interrogatorio
				if (genMsj)
					fprintf(stdout, "\nIniciando la sesion de interrogatorio %d...\n", ii + 1);
				interrogatorio();
				if (genMsj){
					fprintf(stdout, "\nCulminada la sesion de interrogatorio %d\n", ii + 1);
					fprintf(stdout, "\tError cuadratico normalizado de prediccion alcanzado: %.5lf\n", errInterrogatorioNorm);
				}

				//Acumulacion de la suma de ECM de los interrogatorios
				ecmPredPromNorm += errInterrogatorioNorm;

				//Acumulacion de las correlaciones del interrogatorio
				for (jj = 0; jj < noutattr; jj++)
					corrMultIntProm[jj] += corrMultInt[jj];

				//Generar reportes detallados
				if (genReportes){
					//Desnormalizacion de los patrones
					desnormalizar();

					//Generacion de un reporte los resultados de la actual sesion de entrenamiento/interrogatorio
					reporte_entrenamiento_interrogatorio(ii + 1);

					//Generacion de un reporte con los datos de la arquitectura de la red
					reporte_arquitectura(ii + 1);

					//Normalizacion de los patrones
					normalizar();
				}
			}
			//Calculo del ECM de entrenamiento normalizado
			ecmEntrPromNorm /= cantRep;

			//Calculo del ECM promedio de prediccion normalizado
			ecmPredPromNorm /= cantRep;

			//Correlaciones promedio
			for (ii = 0; ii < noutattr; ii++){
				corrMultEntrProm[ii] /= cantRep;
				corrMultIntProm[ii] /= cantRep;
			}

			//Enviar el ECM de prediccion a el proceso padre de donde se ejecuto el algoritmo (solo en caso de que la
			//ejecucion no sea en modo generar reportes)
			if (!genReportes)
				generar_resultados();
			//Generacion de un reporte global de estadisticas
			else{
				reporte_global_corridas();
			}
		}
	}
	//Modo de solo interrogatorio
	else{
		//Se carga una red a partir de una arquitectura ya definida
		cargar_red();

		//Normalizacion de los patrones
		normalizar();

		//Division de patrones
		definir_division(0);

		//Asignar patrones de interrogatorio
		asignar_patrones();

		//Fase de interrogatorio
		if (genMsj)
			fprintf(stdout, "\nIniciando la fase de interrogatorio con la arquitectura dada...\n");
		interrogatorio();
		if (genMsj){
			fprintf(stdout, "\nCulminada la fase de interrogatorio\n");
			fprintf(stdout, "\tError cuadratico normalizado de prediccion alcanzado: %.5lf\n", errInterrogatorioNorm);
		}

		//Desnormalizacion de los patrones
		desnormalizar();

		//Generacion de un reporte con los resultados del interrogatorio
		reporte_interrogatorio();
	}

	//Si se desean reportes, entonces se imprime un mensaje de salida exitosa
	if (genMsj){
		time(&time_fin);
		fprintf(stdout, "\nEl programa ha culminado exitosamente!\n");
		fprintf(stdout, "Tiempo estimado de ejecucion: %.2lfseg\n", difftime(time_fin, time_inicial));
		if (genReportes || modoInterrogatorio){
			if (rutaSalida != NULL)
				fprintf(stdout, "Puede proceder a revisar los reportes guardados en el directorio '%s'\n", rutaSalida);
			else
				fprintf(stdout, "Puede proceder a revisar los reportes guardados en el directorio './'\n");
		}
	}

	//Salida
	liberar_recursos();
	return 0;
}

/************************************************************/
/****************       PIE DEL PROGRAMA      ***************/
/************************************************************/

