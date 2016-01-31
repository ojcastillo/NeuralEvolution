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
	Archivo:		libparallel.h

	Descripcion:	Archivo cabecera de la libreria con funciones para dar soporte de
					paralelismo al sistema NeuralEvolution

	Realizado por:	Orlando Castillo
***/

/*****************************************************************/
/***************   INSTRUCCIONES DE PREPROCESADOR  ***************/
/*****************************************************************/

/** Condicional de definicion simbolica **/
#ifndef LIBPARALLEL_H
#define LIBPARALLEL_H

/** Librerias **/
#include <sys/shm.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>

//Archivo con constantes de la compilacion
#ifdef HAVE_CONFIG_H
	#include <config.h>
#endif

//Inclusion del la libreria profile de MPI
#ifdef HAVE_MPI
	#include <mpi.h>
#endif

//Inclusion del la libreria profile de TAU
#ifdef PROFILE
	#include <TAU.h>
#endif

/** Constantes **/
#define CANT_ESQUEMAS			3							//Cantidad de esquemas paralelos disponibles
#define NOMBRE_ARCH_RESULT		"./ecm_prom.red"			//Nombre del archivo a utilizar para los resultados en el esquema secuencial

/*****************************************************************/
/************   DEFINICION DE TIPOS Y ENUMERADOS   ***************/
/*****************************************************************/

/* Tipos de esquemas de ejecucion que soporta el programa */
typedef enum {FORK = 1, MPI = 2, SECUENCIAL = 3} TIPO_ESQUEMA;

/* Estructura con los datos asociados al esquema fork */
typedef struct{
	int maestro;					//Determina si el proceso es el maestro
	int id_segmento;				//Identificador del segmento de memoria compartida
	double *mem_compartida;			//Apuntador a la zona de memoria compartida
	pid_t *vec_hijos;				//Vector con los identificadores de procesos de cada hijo
} fork_esquema_t;

/* Estructura con los datos asociados al esquema mpi */
#ifdef HAVE_MPI
	typedef struct{
		int maestro;					//Determina si el proceso es el maestro
		MPI_Comm *vec_hijos;			//Comunicadores a los proceso hijos
	} mpi_esquema_t;
#endif

	/*********************************************************************/
	/***************   PROTOTIPOS DE FUNCIONES   *************************/
	/*********************************************************************/

/**
 * Envio del ECM promedio conseguido por la instancia de la red para el esquema secuencial
 *
 * Parametros:
 * 		double err				- Real con el valor del ECM promedio
 *
 * Salida
 * 		int 					- Entero con el valor 1 si fue exitoso el envio, 0 en caso contrario
 */
int enviar_error_secuencial(double err);

/**
 * Recepcion del ECM promedio conseguido por la instancia de la red ejecutada de forma secuencial
 *
 * Parametros:
 * 		double *err				- Referencia a la variable real que almacenara el valor del ECM promedio enviado
 *
 * Salida
 * 		int 					- Entero con el valor 1 si fue exitosa la recepcion, 0 en caso contrario
 */
int recibir_error_secuencial(double *err);

/**
 * Ejecuta de forma secuencial el programa de la red con los parametros especificados
 *
 * Parametros:
 * 		char *comandoRNA			- Comando para la ejecucion de la RNA
 * 		int cant_neu1				- Cantidad de neuronas en la capa oculta 1
 * 		int cant_neu2				- Cantidad de neuronas en la capa oculta 2
 * 		float const_apr				- Valor de la constante de aprendizaje
 * 		float raz_mom				- Valor de la razon de momentum
 * 		int max_iter				- Maxima cantidad de iteraciones para el entrenamiento
 * 		int tipo_fun_o				- Tipo de funcion de activacion a usar por las neuronas en las capas ocultas
 * 		int tipo_fun_s				- Tipo de funcion de activacion a usar por las neuronas en la capa de salida
 * 		int cant_rep				- Cantidad de repeticiones de entrenamiento e interrogatorio
 * 		int gen_rep					- Entero que especifica si se desea la generacion de reportes detallados (1) o no (0)
 * 		const char *ruta			- Ruta al directorio donde se encuentra el archivo con los patrones
 * 		const char *nombre			- Nombre del archivo con los patrones (que se asume tiene sufijo .dat)
 * 		const char *sruta			- Ruta al directorio en donde se almacenran los reportes (si se desean)
 *
 *	Salida
 *		int							- Entero con el valor 1 si pudo ejecutarse el programa correctamente, 0 en caso contrario
 */
int instancia_rna_secuencial(const char *comandoRNA, int cant_neu1, int cant_neu2, float const_apr, float raz_mom, int max_iter,
		int tipo_fun_o, int tipo_fun_s, int cant_rep, int gen_rep, const char *ruta, const char *nombre, const char *sruta);

/**
 * Prepara los datos necesarios para el funcionamiento correcto del esquema paralelo usando fork para el proceso maestro
 *
 * Parametros:
 * 		int tam_segmento 	- Entero con la cantidad de bloques de memoria a reservar para la memoria compartida
 *
 * Salida:
 * 		int					- Entero con el valor 1 si fue exitosa la preparacion, 0 en caso contrario
 */
int preparar_fork_maestro(int tam_segmento);

/**
 * Prepara los datos necesarios para el funcionamiento correcto del esquema paralelo usando fork para un proceso esclavo
 *
 * Parametros:
 * 		int segmento 		- Segmento de memoria al cual se realizara el enlace
 *
 * Salida:
 * 		int					- Entero con el valor 1 si fue exitosa la preparacion, 0 en caso contrario
 */
int preparar_fork_esclavo(int segmento);

/**
 * Limpieza de los datos asociados al esquema paralelo preparado previamente con fork
 *
 * Salida:
 * 		int							- Entero con el valor 1 si fue exitosa la liberacion, 0 en caso contrario
 */
int liberar_fork();

/**
 * Envio del ECM promedio conseguido por la instancia de la red a traves de un segmento de memoria compartida por el esquema fork
 *
 * Parametros:
 * 		int pos_mem				- Entero con la posicion de memoria a escribir (sin sincronizacion)
 * 		double err				- Real con el valor del ECM promedio
 *
 * Salida
 * 		int 					- Entero con el valor 1 si fue exitoso el envio, 0 en caso contrario
 */
int enviar_error_fork(int pos_mem, double err);

/**
 * Recepcion del ECM promedio conseguido por la instancia de la red que le corresponde la posicion de memoria especificada
 *
 * Parametros:
 * 		int	pos_mem				- Entero con la posicion de memoria
 * 		double *err				- Referencia a la variable real que almacenara el valor del ECM promedio enviado
 *
 * Salida
 * 		int 					- Entero con el valor 1 si fue exitosa la recepcion, 0 en caso contrario
 */
int recibir_error_fork(int pos_mem, double *err);

/**
 * Crea una nueva instancia de proceo de RNA de RP con los argumentos proporcionados usando fork
 *
 * Parametros:
 * 		const char *comandoRNA		- Comando para la ejecucion de la RNA
 * 		int cant_neu1				- Cantidad de neuronas en la capa oculta 1
 * 		int cant_neu2				- Cantidad de neuronas en la capa oculta 2
 * 		float const_apr				- Valor de la constante de aprendizaje
 * 		float raz_mom				- Valor de la razon de momentum
 * 		int max_iter				- Maxima cantidad de iteraciones para el entrenamiento
 * 		int tipo_fun_o				- Tipo de funcion de activacion a usar por las neuronas en las capas ocultas
 * 		int tipo_fun_s				- Tipo de funcion de activacion a usar por las neuronas en la capa de salida
 * 		int cant_rep				- Cantidad de repeticiones de entrenamiento e interrogatorio
 * 		const char *ruta			- Ruta al directorio donde se encuentra el archivo con los patrones
 * 		const char *nombre			- Nombre del archivo con los patrones (que se asume tiene sufijo .dat)
 * 		int pos_mem					- Posicion unica de memoria asignada al proceso
 *
 *	Salida
 *		int							- Entero con el valor 1 si pudo crearse la instancia, 0 en caso contrario
 */
int instancia_rna_fork(const char *comandoRNA, int cant_neu1, int cant_neu2, float const_apr, float raz_mom, int max_iter,
		int tipo_fun_o, int tipo_fun_s, int cant_rep, const char *ruta, const char *nombre, int pos_mem);


/**
 * Prepara los recursos necesarios para el funcionamiento correcto del proceso maestro en el esquema con mpi
 *
 * Salida:
 * 		int					- Entero con el valor 1 si fue exitosa la preparacion, 0 en caso contrario
 */
int preparar_mpi_maestro(int cant_procesos);

/**
 * Prepara los recursos necesarios para el funcionamiento correcto de un proceso esclavo en el esquema con mpi
 *
 * Salida:
 * 		int					- Entero con el valor 1 si fue exitosa la preparacion, 0 en caso contrario
 */
int preparar_mpi_esclavo();

/**
 * Libera los recursos necesarios para el funcionamiento correcto del esquema con mpi
 *
 * Salida:
 * 		int					- Entero con el valor 1 si fue exitosa la liberacion, 0 en caso contrario
 */
int liberar_mpi();

/**
 * Envio del ECM promedio conseguido por la instancia de la red que le corresponde la posicion especificada en el vector de
 * comunicadores para el esquema con mpi
 *
 * Parametros:
 * 		int	pos_hijo			- Entero con la posicion de memoria en vector de los comunicadores de hijos
 * 		double err				- Real con el valor del ECM promedio
 *
 * Salida
 * 		int 					- Entero con el valor 1 si fue exitoso el envio, 0 en caso contrario
 */
int enviar_error_mpi(int pos_hijo, double err);

/**
 * Recepcion del ECM promedio conseguido por la instancia de la red que le corresponde la posicion especificada
 *
 * Parametros:
 * 		int	pos_hijo			- Entero con la posicion de memoria en vector de los comunicadores de hijos
 * 		double *err				- Referencia a la variable real que almacenara el valor del ECM promedio enviado
 *
 * Salida
 * 		int 					- Entero con el valor 1 si fue exitosa la recepcion, 0 en caso contrario
 */
int recibir_error_mpi(int pos_hijo, double *err);

/**
 * Crea una nueva instancia de proceo de RNA de RP con los argumentos proporcionados usando mpi
 *
 * Parametros:
 * 		char *comandoRNA		- Comando para la ejecucion de la RNA
 * 		int cant_neu1				- Cantidad de neuronas en la capa oculta 1
 * 		int cant_neu2				- Cantidad de neuronas en la capa oculta 2
 * 		float const_apr				- Valor de la constante de aprendizaje
 * 		float raz_mom				- Valor de la razon de momentum
 * 		int max_iter				- Maxima cantidad de iteraciones para el entrenamiento
 * 		int tipo_fun_o				- Tipo de funcion de activacion a usar por las neuronas en las capas ocultas
 * 		int tipo_fun_s				- Tipo de funcion de activacion a usar por las neuronas en la capa de salida
 * 		int cant_rep				- Cantidad de repeticiones de entrenamiento e interrogatorio
 * 		const char *ruta			- Ruta al directorio donde se encuentra el archivo con los patrones
 * 		const char *nombre			- Nombre del archivo con los patrones (que se asume tiene sufijo .dat)
 * 		int pos_hijo				- Posicion asignada a este proceso en el vector de comunicadores (se asume es unico)
 *
 *	Salida
 *		int							- Entero con el valor 1 si pudo crearse la instancia, 0 en caso contrario
 */
int instancia_rna_mpi(char *comandoRNA, int cant_neu1, int cant_neu2, float const_apr, float raz_mom, int max_iter,
		int tipo_fun_o, int tipo_fun_s, int cant_rep, const char *ruta, const char *nombre, int pos_hijo);


//Fin de la definicion de libparallel.h
#endif
