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
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // 0 = mestre, >0 = trabalhador)
    MPI_Comm_size(MPI_COMM_WORLD, &size); 

    if (argc < 4) {
        if (rank == 0)
            fprintf(stderr, "Uso: %s <square_size> <max_iter> <magnify>\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int square_size = atoi(argv[1]);
    int itermax     = atoi(argv[2]);
    double magnify  = atof(argv[3]);

    int squares_x   = (hxres + square_size - 1) / square_size;
    int squares_y   = (hyres + square_size - 1) / square_size;
    int total_tiles = squares_x * squares_y;

    // ------------------------ Mestre (0) ------------------------------------------
    if (rank == 0) {
        // Aloca a memória pra imagem completa de uma vez
        unsigned char *image = malloc(hxres * hyres * 3);
        if (!image) {
            fprintf(stderr, "Erro alocação imagem\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        } 

        memset(image, 0, hxres * hyres * 3);

        int next_tile      = 0;
        int active_workers = size - 1; // Menos o mestre
        int frame_id       = 0;

        // Distribui o primeiro lote de trabalho pros trabalhadores disponíveis
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

        // Bag of Tasks, quem terminar primeiro, ganha uma task nova
        while (active_workers > 0) {
            MPI_Status status;
            Tile meta;
            
            MPI_Recv(&meta, sizeof(Tile), MPI_BYTE, MPI_ANY_SOURCE, TAG_DONE, MPI_COMM_WORLD, &status);

            int size_buf = meta.w * meta.h * 3;
            unsigned char *buf = malloc(size_buf);
            MPI_Recv(buf, size_buf, MPI_UNSIGNED_CHAR, status.MPI_SOURCE, TAG_DONE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int dy = 0; dy < meta.h; ++dy) { // Percorre linhas do tile pra ajustar as linhas do tile na imagem global
                int dst_offset = ((meta.y + dy) * hxres + meta.x) * 3;
                int src_offset = (dy * meta.w) * 3;
                memcpy(&image[dst_offset], &buf[src_offset], meta.w * 3);
            }
            free(buf);

            char fname[64];
            snprintf(fname, sizeof(fname), "frame_%04d.ppm", frame_id++);
            FILE *fp = fopen(fname, "wb");
            if (fp) {
                fprintf(fp, "P6\n%d %d\n255\n", hxres, hyres);
                fwrite(image, 1, hxres * hyres * 3, fp);
                fclose(fp);
                printf("[Mestre] Frame salvo: %s\n", fname);
            } else {
                fprintf(stderr, "[Erro] Falha ao salvar %s\n", fname);
            }

            // Verifica se ainda tem imagem para processar
            if (next_tile < total_tiles) {
                // Monta o próximo bloco e manda para o trabalhador que acabou de ficar livre
                Tile t = {
                    .x = (next_tile % squares_x) * square_size,
                    .y = (next_tile / squares_x) * square_size
                };
                t.w = (t.x + square_size <= hxres) ? square_size : (hxres - t.x); //calcula largura
                t.h = (t.y + square_size <= hyres) ? square_size : (hyres - t.y); //altura
                
                MPI_Send(&t, sizeof(Tile), MPI_BYTE, status.MPI_SOURCE, TAG_WORK, MPI_COMM_WORLD);
                next_tile++;
            } else {
                // Manda um sinal de parada se a imagem acabou
                MPI_Send(NULL, 0, MPI_BYTE, status.MPI_SOURCE, TAG_STOP, MPI_COMM_WORLD);
                active_workers--;
            }
        }

        // Grava a imagem
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
    
    // ------------------------ Trabalhador (>0) ------------------------------------------
    else {
        // Aloca o buffer de pixels com o tamanho máximo possível do tile uma vez só
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
            
            // Percorre cada pixel do tile recebido
            for (int y = t.y; y < t.y + t.h; ++y) {
                for (int x = t.x; x < t.x + t.w; ++x) {
                    // Mapeando pixel p numero complexo
                    double cx = ((double)x / hxres - 0.5) / magnify * 3.0 - 0.7;
                    double cy = ((double)y / hyres - 0.5) / magnify * 3.0;
                    double zx = 0.0, zy = 0.0, zx2;
                    int it;

                    // Mandelbrot iterativamente
                    for (it = 0; it < itermax; ++it) {
                        zx2 = zx * zx - zy * zy + cx; // parte real
                        zy  = 2.0 * zx * zy + cy; // parte imaginária
                        zx  = zx2; // atualiza real
                        if (zx * zx + zy * zy > 4.0) break;
                    }

                    // Define a cor do pixel dependendo de quão rápido ele escapou
                    if (it < itermax) { // se escapou
                        buf[idx++] = 0;           // RGB               
                        buf[idx++] = (unsigned char)((it * 2) % 50);    
                        buf[idx++] = (unsigned char)((it * 12) % 150); 
                    } else {
                        // Pinta de preto o que ta dentro do conjunto
                        buf[idx++] = 0; buf[idx++] = 0; buf[idx++] = 0;
                    }
                }
            }

            MPI_Send(&t, sizeof(Tile), MPI_BYTE, 0, TAG_DONE, MPI_COMM_WORLD); // envia primeiro o metadado 
            MPI_Send(buf, t.w * t.h * 3, MPI_UNSIGNED_CHAR, 0, TAG_DONE, MPI_COMM_WORLD); // envia dados
        }

        free(buf);
    }

    MPI_Finalize();
    return 0;
}