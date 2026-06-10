#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

const int hxres = 500;
const int hyres = 500;

enum Tags { TAG_WORK = 1, TAG_DONE = 2, TAG_STOP = 3 };

typedef struct {
    int x, y, w, h;
} Tile;

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4) {
        if (rank == 0)
            fprintf(stderr, "Uso: %s <square_size> <max_iter> <magnify>\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int square_size = atoi(argv[1]);
    int itermax = atoi(argv[2]);
    double magnify = atof(argv[3]);

    int squares_x = (hxres + square_size - 1) / square_size;
    int squares_y = (hyres + square_size - 1) / square_size;
    int total_tiles = squares_x * squares_y;

    if (rank == 0) {
        // Aloca a imagem final contígua na memória
        unsigned char *image = malloc(hxres * hyres * 3);
        if (!image) {
            fprintf(stderr, "Erro alocação imagem\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        int next_tile = 0;
        int active_workers = size - 1;

        // Envia tarefas iniciais para os processos trabalhadores disponíveis
        for (int dest = 1; dest < size && next_tile < total_tiles; ++dest) {
            Tile t = {
                .x = (next_tile % squares_x) * square_size,
                .y = (next_tile / squares_x) * square_size
            };
            t.w = (t.x + square_size <= hxres) ? square_size : (hxres - t.x);
            t.h = (t.y + square_size <= hyres) ? square_size : (hyres - t.y);
            MPI_Send(&t, sizeof(Tile), MPI_BYTE, dest, TAG_WORK, MPI_COMM_WORLD);
            next_tile++;
        }

        // Recebe os resultados dos blocos computados e despacha novos trabalhos
        while (active_workers > 0) {
            MPI_Status status;
            Tile meta;
            MPI_Recv(&meta, sizeof(Tile), MPI_BYTE, MPI_ANY_SOURCE, TAG_DONE, MPI_COMM_WORLD, &status);

            int size_buf = meta.w * meta.h * 3;
            unsigned char *buf = malloc(size_buf);
            MPI_Recv(buf, size_buf, MPI_UNSIGNED_CHAR, status.MPI_SOURCE, TAG_DONE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Cópia eficiente dos blocos para o buffer da imagem principal usando memcpy
            for (int dy = 0; dy < meta.h; ++dy) {
                int dst_offset = ((meta.y + dy) * hxres + meta.x) * 3;
                int src_offset = (dy * meta.w) * 3;
                memcpy(&image[dst_offset], &buf[src_offset], meta.w * 3);
            }
            free(buf);

            // Envia a próxima tarefa da fila ou comando de finalização
            if (next_tile < total_tiles) {
                Tile t = {
                    .x = (next_tile % squares_x) * square_size,
                    .y = (next_tile / squares_x) * square_size
                };
                t.w = (t.x + square_size <= hxres) ? square_size : (hxres - t.x);
                t.h = (t.y + square_size <= hyres) ? square_size : (hyres - t.y);
                MPI_Send(&t, sizeof(Tile), MPI_BYTE, status.MPI_SOURCE, TAG_WORK, MPI_COMM_WORLD);
                next_tile++;
            } else {
                // Sem mais tarefas pendentes: sinaliza parada para o worker livre
                MPI_Send(NULL, 0, MPI_BYTE, status.MPI_SOURCE, TAG_STOP, MPI_COMM_WORLD);
                active_workers--;
            }
        }

        // Escreve uma única vez do arquivo ppm final reduzindo I/O
        printf("[Mestre] Gravando arquivo de imagem final...\n");
        FILE *fp = fopen("mandelbrot_final.ppm", "wb");
        if (fp) {
            fprintf(fp, "P6\n%d %d\n255\n", hxres, hyres);
            fwrite(image, 1, hxres * hyres * 3, fp);
            fclose(fp);
            printf("[Mestre] Arquivo 'mandelbrot_final.ppm' gerado.\n");
        } else {
            fprintf(stderr, "[Erro] Falha ao abrir o arquivo para escrita.\n");
        }

        free(image);
    } 
    else {
        // Aloca o buffer máximo uma única vez fora do laço
        int max_buf_size = square_size * square_size * 3;
        unsigned char *buf = malloc(max_buf_size);
        if (!buf) {
            fprintf(stderr, "Erro alocação buf no rank %d\n", rank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        while (1) {
            MPI_Status status;
            Tile t;
            MPI_Recv(&t, sizeof(Tile), MPI_BYTE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == TAG_STOP) break;

            int idx = 0;
            for (int y = t.y; y < t.y + t.h; ++y) {
                for (int x = t.x; x < t.x + t.w; ++x) {
                    double cx = ((double)x / hxres - 0.5) / magnify * 3.0 - 0.7;
                    double cy = ((double)y / hyres - 0.5) / magnify * 3.0;
                    double zx = 0.0, zy = 0.0, zx2;
                    int it;
                    
                    for (it = 0; it < itermax; ++it) {
                        zx2 = zx * zx - zy * zy + cx;
                        zy  = 2.0 * zx * zy + cy;
                        zx  = zx2;
                        if (zx * zx + zy * zy > 4.0) break;
                    }
                    
                    if (it < itermax) {
                        buf[idx++] = 0;                                 
                        buf[idx++] = (unsigned char)((it * 2) % 50);    
                        buf[idx++] = (unsigned char)((it * 12) % 150);
                    } else {
                        // Interior do conjunto de Mandelbrot (Cor escura/Preto)
                        buf[idx++] = 0; buf[idx++] = 0; buf[idx++] = 0;
                    }
                }
            }

            // Envia o cabeçalho de metadados e os dados processados de volta ao Mestre
            MPI_Send(&t, sizeof(Tile), MPI_BYTE, 0, TAG_DONE, MPI_COMM_WORLD);
            MPI_Send(buf, t.w * t.h * 3, MPI_UNSIGNED_CHAR, 0, TAG_DONE, MPI_COMM_WORLD);
        }

        // Libera a memória uma única vez ao encerrar o processo trabalhador
        free(buf);
    }

    MPI_Finalize();
    return 0;
}