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

	Descripcion:	Archivo fuente de la libreria con funciones para dar soporte de
					paralelismo al sistema NeuralEvolution

	Realizado por:	Orlando Castillo
***/

	/*****************************************************************/
	/***************   INSTRUCCIONES DE PREPROCESADOR  ***************/
	/*****************************************************************/

/** Librerias **/
#include "../include/libparallel.h"

	/****************************************************************/
	/***************   VARIABLES GLOBALES   *************************/
	/****************************************************************/

/** Variables para el esquema paralelo con memoria compartida de fork **/
fork_esquema_t datos_fork;		//Estructura con los datos necesarios del fork
int esquema_;					//Esquema seleccionado
int preparado_ = 0;				//Indica que los datos de un esquema fueron previamente preparados

#ifdef HAVE_MPI
	mpi_esquema_t datos_mpi;		//Estructura con los datos necesarios de mpi
#endif

	/*************************************************************************/
	/***************   IMPLEMENTACION DE FUNCIONES   *************************/
	/*************************************************************************/

/**
 * Envio del ECM promedio conseguido por la instancia de la red para el esquema secuencial
 *
 * Parametros:
 * 		double err				- Real con el valor del ECM promedio
 *
 * Salida
 * 		int 					- Entero con el valor 1 si fue exitoso el envio, 0 en caso contrario
 */
int enviar_error_secuencial(double err){
	//Variables
	int result = 0;		//Indica si los datos fueron enviados correctamente
	FILE *out;			//Archivo en donde se escribiran los datos

	//El archivo en donde se escribiran los resultados se creo exitosamente
	if ((out = fopen(NOMBRE_ARCH_RESULT, "w")) != NULL){
		fprintf(out, "%.10lf\n", err);
		result = 1;
		fclose(out);
	}

	//Resultado
	return result;
}

/**
 * Recepcion del ECM promedio conseguido por la instancia de la red ejecutada de forma secuencial
 *
 * Parametros:
 * 		double *err				- Referencia a la variable real que almacenara el valor del ECM promedio enviado
 *
 * Salida
 * 		int 					- Entero con el valor 1 si fue exitosa la recepcion, 0 en caso contrario
 */
int recibir_error_secuencial(double *err){
	//Variables
	int result = 0;		//Indica si los datos fueron enviados correctamente
	FILE *in;			//Archivo en donde se leeran los datos

	//El archivo en donde se escribiran los resultados se creo exitosamente
	if ((in = fopen(NOMBRE_ARCH_RESULT, "r")) != NULL){
		if (fscanf(in, "%lf", err) != EOF){
			result = 1;
			remove(NOMBRE_ARCH_RESULT);
		}
		fclose(in);
	}

	//Resultado
	return result;
}

/**
 * Ejecuta de forma secuencial el programa de la red con los parametros especificados
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
 * 		int gen_rep					- Entero que especifica si se desea la generacion de reportes detallados (1) o no (0)
 * 		const char *ruta			- Ruta al directorio donde se encuentra el archivo con los patrones
 * 		const char *nombre			- Nombre del archivo con los patrones (que se asume tiene sufijo .dat)
 * 		const char *sruta			- Ruta al directorio en donde se almacenran los reportes (si se desean)
 *
 *	Salida
 *		int							- Entero con el valor 1 si pudo ejecutarse el programa correctamente, 0 en caso contrario
 */
int instancia_rna_secuencial(const char *comandoRNA, int cant_neu1, int cant_neu2, float const_apr, float raz_mom, int max_iter,
		int tipo_fun_o, int tipo_fun_s, int cant_rep, int gen_rep, const char *ruta, const char *nombre, const char *sruta){
	//Variables
	char *comando;			//Comando a ejecutar
	int tam_int = 0;		//Cantidad maxima de bits en un int
	int num = INT_MAX;		//Valor auxiliar utilizado para determinar la cantidad maxima de bits
	int result = 0;			//Indica si el comando fue ejecutado exitosamente

	//Cantidad maxima de bits de un entero
	do{
		tam_int++;
		num /= 10;
	}while (num > 0);

	//Reserva de espacio de memoria
	comando = (char *) calloc((tam_int * 9) + ((tam_int + 8) * 2) + strlen(ruta) + strlen(nombre)
			+ strlen(sruta) + 100, sizeof(char));

	//Creacion de la cadena del comando
	sprintf(comando, "'%s' --debug=n --mensajes=n 1 %d %d %.5f %.3f %d %d %d %d '%s' %s", comandoRNA, cant_neu1, cant_neu2,
			const_apr, raz_mom, max_iter, tipo_fun_o, tipo_fun_s, cant_rep, ruta, nombre);
	if (gen_rep){
		sprintf(comando, "%s 0 '%s'",comando, sruta);
	}
	else
		strcat(comando, " 1");

	//Ejecucion del comando
	if (system(comando) == 0)
		result = 1;

	//Liberacion de la cadena
	free(comando);

	//Resultado
	return result;
}

/**
 * Prepara los datos necesarios para el funcionamiento correcto del esquema paralelo usando fork para el proceso maestro
 *
 * Parametros:
 * 		int tam_segmento 	- Entero con la cantidad de bloques de memoria a reservar para la memoria compartida
 *
 * Salida:
 * 		int					- Entero con el valor 1 si fue exitosa la preparacion, 0 en caso contrario
 */
int preparar_fork_maestro(int tam_segmento){
	//Variables
	int result;			//Indica si la preparacion fue exitosa
	void *adrs;			//Direccion retornada para el enlace

	//Ya fue preparado anteriormente
	if (preparado_)
		return 0;

	//Preparar la zona de memoria compartida
	datos_fork.id_segmento = shmget (IPC_PRIVATE, tam_segmento * sizeof(double), IPC_CREAT | IPC_EXCL | S_IRUSR |
					S_IWUSR |S_IROTH | S_IWOTH);

	//Se asume el fallo de la operacion
	result = 0;
	preparado_ = 0;

	//Preparacion exitosa
	if (datos_fork.id_segmento != -1){
		//Se intenta enlazar la zona de memoria compartida a un apuntador
		adrs = shmat (datos_fork.id_segmento, 0, 0);

		//El enlace fue exitoso
		if ( *((int *) adrs) != -1){
			//Se enlaza la variable correspondiente
			datos_fork.mem_compartida = (double *) adrs;

			//Reserva de espacio de memoria para el almacenamiento de identificadores de hijos
			datos_fork.vec_hijos = (pid_t *) calloc(tam_segmento, sizeof(pid_t));

			//Exitosa preparacion
			datos_fork.maestro = 1;
			esquema_ = FORK;
			result = 1;
			preparado_ = 1;
		}
	}

	//Resultado
	return  result;
}

/**
 * Prepara los datos necesarios para el funcionamiento correcto del esquema paralelo usando fork para un proceso esclavo
 *
 * Parametros:
 * 		int segmento 		- Segmento de memoria al cual se realizara el enlace
 *
 * Salida:
 * 		int					- Entero con el valor 1 si fue exitosa la preparacion, 0 en caso contrario
 */
int preparar_fork_esclavo(int segmento){
	//Variables
	int result;			//Indica si la preparacion fue exitosa
	void *adrs;			//Direccion retornada para el enlace

	//Ya fue preparado anteriormente
	if (preparado_)
		return 0;

	//Segmento de memoria ya creado
	datos_fork.id_segmento = segmento;

	//Se asume el fallo de la operacion
	result = 0;
	preparado_ = 0;

	//Preparacion exitosa
	if (datos_fork.id_segmento != -1){
		//Se intenta enlazar la zona de memoria compartida a un apuntador
		adrs = shmat (datos_fork.id_segmento, 0, 0);

		//El enlace fue exitoso
		if ( *((int *) adrs) != -1){
			//Se enlaza la variable correspondiente
			datos_fork.mem_compartida = (double *) adrs;

			//El proceso esclavo no tendra procesos hijos
			datos_fork.vec_hijos = NULL;

			//Exitosa preparacion
			datos_fork.maestro = 0;
			esquema_ = FORK;
			result = 1;
			preparado_ = 1;
		}
	}

	//Resultado
	return  result;
}

/**
 * Limpieza de los datos asociados al esquema paralelo preparado previamente con fork
 *
 * Salida:
 * 		int							- Entero con el valor 1 si fue exitosa la liberacion, 0 en caso contrario
 */
int liberar_fork(){
	//Variables
	int result = 0;		//Indica si los datos fueron liberados correctamente

	//Se realiza solo si fue preparado previamente y con esquema fork
	if (preparado_ && esquema_ == FORK){
		//Desenlazar la memoria compartida (Verificando que sea exitosa)
		if (shmdt (datos_fork.mem_compartida) != -1){
			//Desenlace exitoso
			result = 1;
			preparado_ = 0;

			//Las siguientes instrucciones solo se deben realizar en caso de que el proceso sea el maestro
			if (datos_fork.maestro){
				//Liberacion del vector con los identificadores de procesos
				free(datos_fork.vec_hijos);

				//Se liberan todos los recursos asociados al segmento de memoria compartida
				shmctl(datos_fork.id_segmento, IPC_RMID, 0);
			}
		}
	}

	//Resultado
	return result;
}

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
int enviar_error_fork(int pos_mem, double err){
	//Variables
	int result = 0;		//Indica si los datos fueron enviados correctamente

	//Se realiza solo si fue preparado previamente y con esquema fork
	if (preparado_ && esquema_ == FORK){
		//Almacenamiento del ECM promedio en la memoria compartida
		datos_fork.mem_compartida[pos_mem] = err;
		result = 1;
	}

	//Resultado
	return result;
}


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
int recibir_error_fork(int pos_mem, double *err){
	//Variables
	int result = 0;		//Indica si los datos fueron recibidos correctamente
	int estado_hijo;	//Estado retornado por el proceso

	//Se realiza solo si fue preparado previamente y con esquema fork
	if (preparado_ && esquema_ == FORK){
		//Se espera la culminacion del proceso hijo asociado a la posicion de memoria
		waitpid(datos_fork.vec_hijos[pos_mem], &estado_hijo, WUNTRACED);

		//El proceso hijo culmino de forma correcta
		if (WIFEXITED (estado_hijo)){
			//El proceso hijo culmino de forma exitosa, por lo que se procede a obtener el error
			if (WEXITSTATUS (estado_hijo) == 0){
				*err = datos_fork.mem_compartida[pos_mem];
				result = 1;
			}
		}
	}

	//Resultado
	return result;
}

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
		int tipo_fun_o, int tipo_fun_s, int cant_rep, const char *ruta, const char *nombre, int pos_mem){
	//Variables
	pid_t pid;				//Identificador de proceso retornado al crear la instancia
	int result = 0;			//Indica si la instancia fue creada exitosamente
	char **params;			//Cadena de caracteres con los parametros
	int tam_int = 0;		//Cantidad maxima de bits en un int
	int num = INT_MAX;		//Valor auxiliar utilizado para determinar la cantidad maxima de bits

	//Se verifican que los datos se hayan preparado previamente
	if (preparado_ && esquema_ == FORK){
		//Creacion de un proceso hijo a partir del proceso actual
		pid = fork();

		//Creacion no exitosa
		if (pid == -1)
			result = 0;
		//Condicion que solo aplica al proceso hijo
		else if (pid == 0){
			//Cantidad maxima de bits de un entero
			do{
				tam_int++;
				num /= 10;
			}while (num > 0);

			//Conversion del valor de los parametros a cadenas de caracteres para la ejecucion del proceso
			params = calloc(18, sizeof(char *));
			params[0] = "RN_BP";
			params[1] = "--debug=n";
			params[2] = "--mensajes=n";
			params[3] = "1";
			params[4] = (char *) calloc(tam_int, sizeof(char)); sprintf(params[4], "%d", cant_neu1);
			params[5] = (char *) calloc(tam_int, sizeof(char)); sprintf(params[5], "%d", cant_neu2);
			params[6] = (char *) calloc(tam_int + 8, sizeof(char)); sprintf(params[6], "%.5f", const_apr);
			params[7] = (char *) calloc(tam_int + 8, sizeof(char)); sprintf(params[7], "%.3f", raz_mom);
			params[8] = (char *) calloc(tam_int, sizeof(char)); sprintf(params[8], "%d", max_iter);
			params[9] = (char *) calloc(tam_int, sizeof(char)); sprintf(params[9], "%d", tipo_fun_o);
			params[10] = (char *) calloc(tam_int, sizeof(char)); sprintf(params[10], "%d", tipo_fun_s);
			params[11] = (char *) calloc(tam_int, sizeof(char)); sprintf(params[11], "%d", cant_rep);
			params[12] = (char *) calloc(strlen(ruta) + 5, sizeof(char)); sprintf(params[12], "%s", ruta);
			params[13] = (char *) calloc(strlen(nombre) + 5, sizeof(char)); sprintf(params[13], "%s", nombre);
			params[14] = (char *) calloc(tam_int, sizeof(char)); sprintf(params[14], "2");
			params[15] = (char *) calloc(tam_int, sizeof(char)); sprintf(params[15], "%d", datos_fork.id_segmento);
			params[16] = (char *) calloc(tam_int, sizeof(char)); sprintf(params[16], "%d", pos_mem);
			params[17] = NULL;

			//El hijo crea un nuevo proceso en donde se ejecutara la RNA con los parametros proporcionados
			execvp(comandoRNA, params);

			//Si llega hasta aca, algo malo ocurrio y no se pudo instanciar la red
			exit(-1);
		}
		//Almacenamiento del indentificador del proceso en el proceso padre
		else{
			datos_fork.vec_hijos[pos_mem] = pid;
			result = 1;
		}
	}

	//Resultado
	return result;
}


/**
 * Prepara los recursos necesarios para el funcionamiento correcto del proceso maestro en el esquema con mpi
 *
 * Parametros:
 * 		int cant_procesos	- Maxima cantidad de procesos que se llegaran a crear
 *
 * Salida:
 * 		int					- Entero con el valor 1 si fue exitosa la preparacion, 0 en caso contrario
 */
int preparar_mpi_maestro(int cant_procesos){
	//Variables
	int result = 0;			//Indica si la preparacion fue exitosa

	//Ya fue preparado anteriormente
	if (preparado_)
		return 0;

	//Solo se intentara preparar el esquema si se cuenta con la liberia de mpi
	#ifdef HAVE_MPI
		//Se registra el esquema mpi si fue exitosa la creacion
		if (MPI_Init(NULL, NULL) == MPI_SUCCESS){
			datos_mpi.maestro = 1;
			datos_mpi.vec_hijos = (MPI_Comm *) calloc(cant_procesos, sizeof(MPI_Comm));
			result = 1;
			preparado_ = 1;
			esquema_ = MPI;
		}
	#endif

	//Resultado
	return  result;
}

/**
 * Prepara los recursos necesarios para el funcionamiento correcto de un proceso esclavo en el esquema con mpi
 *
 * Salida:
 * 		int					- Entero con el valor 1 si fue exitosa la preparacion, 0 en caso contrario
 */
int preparar_mpi_esclavo(){
	//Variables
	int result = 0;			//Indica si la preparacion fue exitosa

	//Ya fue preparado anteriormente
	if (preparado_)
		return 0;

	//Solo se intentara preparar el esquema si se cuenta con la liberia de mpi
	#ifdef HAVE_MPI
		//Se registra el esquema mpi si fue exitosa la creacion
		if (MPI_Init(NULL, NULL) == MPI_SUCCESS){
			datos_mpi.maestro = 0;
			datos_mpi.vec_hijos = NULL;
			result = 1;
			preparado_ = 1;
			esquema_ = MPI;
		}
	#endif

	//Resultado
	return  result;
}

/**
 * Libera los recursos necesarios para el funcionamiento correcto del esquema con mpi
 *
 * Salida:
 * 		int					- Entero con el valor 1 si fue exitosa la liberacion, 0 en caso contrario
 */
int liberar_mpi(){
	//Variables
	int result = 0;			//Indica si los datos fueron liberados correctamente

	//Se realiza solo si fue preparado previamente y con esquema mpi
	if (preparado_ && esquema_ == MPI){
		#ifdef HAVE_MPI
			//Liberar los recursos creados para el esquema mpi (Verificando que sea exitosa la liberacion)
			if (MPI_Finalize() == MPI_SUCCESS){
				//Liberacion exitosa
				result = 1;
				preparado_ = 0;

				//Las siguientes instrucciones solo se deben realizar en caso de que el proceso sea el maestro
				if (datos_mpi.maestro){
					//Liberacion del vector con los comunicadores de procesos
					free(datos_mpi.vec_hijos);
				}
			}
		#endif
	}

	//Resultado
	return result;
}

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
int enviar_error_mpi(int pos_hijo, double err){
	//Variables
	int result = 0;			//Indica si los datos fueron enviados correctamente

	//Se realiza solo si fue preparado previamente y con esquema mpi
	if (preparado_ && esquema_ == MPI){
		#ifdef HAVE_MPI
			MPI_Comm padre;			//Comunicador que permite el paso de mensajes entre el proceso esclavo y el proceso maestro padre

			//Se obtiene el comunicador al padre
			MPI_Comm_get_parent(&padre);

			//Envio exitoso del ECM promedio
			if (MPI_Send(&err, 1, MPI_DOUBLE, 0, 1, padre) == MPI_SUCCESS)
				result = 1;
		#endif
	}

	//Resultado
	return result;
}

/**
 * Recepcion del ECM promedio conseguido por la instancia de la red que le corresponde la posicion especificada en el vector de
 * comunicadores para el esquema con mpi
 *
 * Parametros:
 * 		int	pos_hijo			- Entero con la posicion de memoria en vector de los comunicadores de hijos
 * 		double *err				- Referencia a la variable real que almacenara el valor del ECM promedio enviado
 *
 * Salida
 * 		int 					- Entero con el valor 1 si fue exitosa la recepcion, 0 en caso contrario
 */
int recibir_error_mpi(int pos_hijo, double *err){
	//Variables
	int result = 0;			//Indica si los datos fueron recibidos correctamente

	//Se realiza solo si fue preparado previamente y con esquema mpi
	if (preparado_ && esquema_ == MPI){
		#ifdef HAVE_MPI
			//Recepcion exitosa
			if (MPI_Recv(err, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 1, datos_mpi.vec_hijos[pos_hijo],
					MPI_STATUS_IGNORE) == MPI_SUCCESS)
				result = 1;
		#endif
	}

	//Resultado
	return result;
}

/**
 * Crea una nueva instancia de proceo de RNA de RP con los argumentos proporcionados usando mpi
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
 * 		int pos_hijo				- Posicion asignada a este proceso en el vector de comunicadores (se asume es unico)
 *
 *	Salida
 *		int							- Entero con el valor 1 si pudo crearse la instancia, 0 en caso contrario
 */
int instancia_rna_mpi(char *comandoRNA, int cant_neu1, int cant_neu2, float const_apr, float raz_mom, int max_iter,
		int tipo_fun_o, int tipo_fun_s, int cant_rep, const char *ruta, const char *nombre, int pos_hijo){
	//Variables
	int result = 0;			//Indica si la instancia fue creada exitosamente

	//Se verifican que los datos se hayan preparado previamente
	if (preparado_ && esquema_ == MPI){
		#ifdef HAVE_MPI
			int ii;					//Contador
			char **params;			//Cadena de caracteres con los parametros
			int tam_int = 0;		//Cantidad maxima de bits en un int
			int num = INT_MAX;		//Valor auxiliar utilizado para determinar la cantidad maxima de bits

			//Cantidad maxima de bits de un entero
			do{
				tam_int++;
				num /= 10;
			}while (num > 0);

			//Conversion del valor de los parametros a cadenas de caracteres para la ejecucion del proceso
			params = calloc(15, sizeof(char *));
			params[0] = (char *) calloc(20, sizeof(char)); sprintf(params[0], "--debug=n");
			params[1] = (char *) calloc(20, sizeof(char)); sprintf(params[1], "--mensajes=n");
			params[2] = (char *) calloc(tam_int, sizeof(char)); sprintf(params[2], "1");
			params[3] = (char *) calloc(tam_int, sizeof(char)); sprintf(params[3], "%d", cant_neu1);
			params[4] = (char *) calloc(tam_int, sizeof(char)); sprintf(params[4], "%d", cant_neu2);
			params[5] = (char *) calloc(tam_int + 8, sizeof(char)); sprintf(params[5], "%.5f", const_apr);
			params[6] = (char *) calloc(tam_int + 8, sizeof(char)); sprintf(params[6], "%.3f", raz_mom);
			params[7] = (char *) calloc(tam_int, sizeof(char)); sprintf(params[7], "%d", max_iter);
			params[8] = (char *) calloc(tam_int, sizeof(char)); sprintf(params[8], "%d", tipo_fun_o);
			params[9] = (char *) calloc(tam_int, sizeof(char)); sprintf(params[9], "%d", tipo_fun_s);
			params[10] = (char *) calloc(tam_int, sizeof(char)); sprintf(params[10], "%d", cant_rep);
			params[11] = (char *) calloc(strlen(ruta) + 5, sizeof(char)); sprintf(params[11], "%s", ruta);
			params[12] = (char *) calloc(strlen(nombre) + 5, sizeof(char)); sprintf(params[12], "%s", nombre);
			params[13] = (char *) calloc(tam_int, sizeof(char)); sprintf(params[13], "3");
			params[14] = NULL;

			//Se crea de forma dinamica un nuevo proceso que ejecutara la RNA con los parametros dados
			if (MPI_Comm_spawn(comandoRNA, params, 1, MPI_INFO_NULL, 0, MPI_COMM_SELF, &(datos_mpi.vec_hijos[pos_hijo]),
							  MPI_ERRCODES_IGNORE) == MPI_SUCCESS){
				result = 1;
			}

			//Liberacion de memoria del vector de parametros
			for (ii = 0; ii < 13; ii++)
				free(params[ii]);
			free(params);
		#endif
	}

	//Resultado
	return result;
}
