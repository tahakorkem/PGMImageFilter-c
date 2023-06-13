#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>

#define MAX_RES 1024
#define MAX_K 7

typedef struct PGMImage {
    int width;
    int height;
    int maxValue;
    int data[MAX_RES][MAX_RES];
} PGMImage;

char *concat(const char *, const char *);

char *concat_three(const char *, const char *, const char *);

void readPGMFile(char[], PGMImage *);

void savePGMFile(PGMImage *, char[]);

void normalize(PGMImage *);

PGMImage *conv(PGMImage *, int, double[MAX_K][MAX_K], int);

PGMImage *euclideanDistance(PGMImage *, PGMImage *, int);

void applySobel(PGMImage *img, const char *, const char *);

void applyLaplacian(PGMImage *img, const char *, const char *);

int main() {
    // read file
    PGMImage *img = calloc(1, sizeof(PGMImage));
    char *filename = "coins.ascii|fruit|lena"; // pick the image what you want to apply filters
    char *path = concat_three("images/", filename, ".pgm");
    readPGMFile(path, img);

    double kernel_gauss_3x3_sigma1[MAX_K][MAX_K] = {
            {0.0751136079541115, 0.123841403152974, 0.0751136079541115},
            {0.123841403152974,  0.204179955571658, 0.123841403152974},
            {0.0751136079541115, 0.123841403152974, 0.0751136079541115}
    };
    double kernel_gauss_3x3_sigma2[MAX_K][MAX_K] = {
            {0.101868064419816, 0.115431639614227, 0.101868064419816},
            {0.115431639614227, 0.130801183863828, 0.115431639614227},
            {0.101868064419816, 0.115431639614227, 0.101868064419816}
    };
    double kernel_gauss_3x3_sigma4[MAX_K][MAX_K] = {
            {0.108796548095085, 0.112250121255763, 0.108796548095085},
            {0.112250121255763, 0.115813322596608, 0.112250121255763},
            {0.108796548095085, 0.112250121255763, 0.108796548095085}
    };
    double kernel_gauss_5x5_sigma1[MAX_K][MAX_K] = {
            {0.0029690167439505, 0.0133062098910137, 0.0219382312797146, 0.0133062098910137, 0.0029690167439505},
            {0.0133062098910137, 0.0596342954361801, 0.0983203313488458, 0.0596342954361801, 0.0133062098910137},
            {0.0219382312797146, 0.0983203313488458, 0.162102821637127,  0.0983203313488458, 0.0219382312797146},
            {0.0133062098910137, 0.0596342954361801, 0.0983203313488458, 0.0596342954361801, 0.0133062098910137},
            {0.0029690167439505, 0.0133062098910137, 0.0219382312797146, 0.0133062098910137, 0.0029690167439505}
    };
    double kernel_gauss_5x5_sigma2[MAX_K][MAX_K] = {
            {0.0232468398782944, 0.0338239524399223, 0.0383275593839039, 0.0338239524399223, 0.0232468398782944},
            {0.0338239524399223, 0.0492135604085414, 0.0557662698468495, 0.0492135604085414, 0.0338239524399223},
            {0.0383275593839039, 0.0557662698468495, 0.0631914624102647, 0.0557662698468495, 0.0383275593839039},
            {0.0338239524399223, 0.0492135604085414, 0.0557662698468495, 0.0492135604085414, 0.0338239524399223},
            {0.0232468398782944, 0.0338239524399223, 0.0383275593839039, 0.0338239524399223, 0.0232468398782944}
    };
    double kernel_gauss_5x5_sigma4[MAX_K][MAX_K] = {
            {0.0352039519019332, 0.0386639772540046, 0.0398913036395145, 0.0386639772540046, 0.0352039519019332},
            {0.0386639772540046, 0.0424640716832731, 0.0438120260147863, 0.0424640716832731, 0.0386639772540046},
            {0.0398913036395145, 0.0438120260147863, 0.045202769009935,  0.0438120260147863, 0.0398913036395145},
            {0.0386639772540046, 0.0424640716832731, 0.0438120260147863, 0.0424640716832731, 0.0386639772540046},
            {0.0352039519019332, 0.0386639772540046, 0.0398913036395145, 0.0386639772540046, 0.0352039519019332}
    };
    double kernel_gauss_7x7_sigma1[MAX_K][MAX_K] = {
            {1.96519161240319e-05, 0.00023940934949727, 0.00107295826497866, 0.00176900911404382, 0.00107295826497866, 0.00023940934949727, 1.96519161240319e-05},
            {0.00023940934949727,  0.00291660295438644, 0.0130713075831894,  0.0215509428482683,  0.0130713075831894,  0.00291660295438644, 0.00023940934949727},
            {0.00107295826497866,  0.0130713075831894,  0.058581536330607,   0.0965846250185641,  0.058581536330607,   0.0130713075831894,  0.00107295826497866},
            {0.00176900911404382,  0.0215509428482683,  0.0965846250185641,  0.159241125690702,   0.0965846250185641,  0.0215509428482683,  0.00176900911404382},
            {0.00107295826497866,  0.0130713075831894,  0.058581536330607,   0.0965846250185641,  0.058581536330607,   0.0130713075831894,  0.00107295826497866},
            {0.00023940934949727,  0.00291660295438644, 0.0130713075831894,  0.0215509428482683,  0.0130713075831894,  0.00291660295438644, 0.00023940934949727},
            {1.96519161240319e-05, 0.00023940934949727, 0.00107295826497866, 0.00176900911404382, 0.00107295826497866, 0.00023940934949727, 1.96519161240319e-05}
    };
    double kernel_gauss_7x7_sigma2[MAX_K][MAX_K] = {
            {0.00492233115934352, 0.0091961252895862, 0.0133802833441012, 0.0151618473729641, 0.0133802833441012, 0.0091961252895862, 0.00492233115934352},
            {0.0091961252895862,  0.0171806238963096, 0.0249976602669148, 0.0283260600617446, 0.0249976602669148, 0.0171806238963096, 0.0091961252895862},
            {0.0133802833441012,  0.0249976602669148, 0.0363713810739036, 0.0412141741997979, 0.0363713810739036, 0.0249976602669148, 0.0133802833441012},
            {0.0151618473729641,  0.0283260600617446, 0.0412141741997979, 0.0467017777389277, 0.0412141741997979, 0.0283260600617446, 0.0151618473729641},
            {0.0133802833441012,  0.0249976602669148, 0.0363713810739036, 0.0412141741997979, 0.0363713810739036, 0.0249976602669148, 0.0133802833441012},
            {0.0091961252895862,  0.0171806238963096, 0.0249976602669148, 0.0283260600617446, 0.0249976602669148, 0.0171806238963096, 0.0091961252895862},
            {0.00492233115934352, 0.0091961252895862, 0.0133802833441012, 0.0151618473729641, 0.0133802833441012, 0.0091961252895862, 0.00492233115934352}
    };
    double kernel_gauss_7x7_sigma4[MAX_K][MAX_K] = {
            {0.0147600268537462, 0.0172562196606719, 0.0189522496312038, 0.0195538586142718, 0.0189522496312038, 0.0172562196606719, 0.0147600268537462},
            {0.0172562196606719, 0.0201745647164444, 0.0221574246402495, 0.0228607767997356, 0.0221574246402495, 0.0201745647164444, 0.0172562196606719},
            {0.0189522496312038, 0.0221574246402495, 0.0243351702298765, 0.0251076514550435, 0.0243351702298765, 0.0221574246402495, 0.0189522496312038},
            {0.0195538586142718, 0.0228607767997356, 0.0251076514550435, 0.0259046538665264, 0.0251076514550435, 0.0228607767997356, 0.0195538586142718},
            {0.0189522496312038, 0.0221574246402495, 0.0243351702298765, 0.0251076514550435, 0.0243351702298765, 0.0221574246402495, 0.0189522496312038},
            {0.0172562196606719, 0.0201745647164444, 0.0221574246402495, 0.0228607767997356, 0.0221574246402495, 0.0201745647164444, 0.0172562196606719},
            {0.0147600268537462, 0.0172562196606719, 0.0189522496312038, 0.0195538586142718, 0.0189522496312038, 0.0172562196606719, 0.0147600268537462}
    };

    PGMImage *imgNew;

    // apply sobel filter to original image
    applySobel(img, filename, "_original");

    // apply gauss filter to original image
    imgNew = conv(img, 3, kernel_gauss_3x3_sigma1, 0);
    savePGMFile(imgNew, concat(filename, "_gauss_3x3_sigma1.pgm"));
    applySobel(img, filename, "_gauss_3x3_sigma1");
    applyLaplacian(imgNew, filename, "_gauss_3x3_sigma1");
    free(imgNew);

    imgNew = conv(img, 3, kernel_gauss_3x3_sigma2, 0);
    savePGMFile(imgNew, concat(filename, "_gauss_3x3_sigma2.pgm"));
    applySobel(imgNew, filename, "_gauss_3x3_sigma2");
    applyLaplacian(imgNew, filename, "_gauss_3x3_sigma2");
    free(imgNew);

    imgNew = conv(img, 3, kernel_gauss_3x3_sigma4, 0);
    savePGMFile(imgNew, concat(filename, "_gauss_3x3_sigma4.pgm"));
    applySobel(imgNew, filename, "_gauss_3x3_sigma4");
    applyLaplacian(imgNew, filename, "_gauss_3x3_sigma4");
    free(imgNew);

    imgNew = conv(img, 5, kernel_gauss_5x5_sigma1, 0);
    savePGMFile(imgNew, concat(filename, "_gauss_5x5_sigma1.pgm"));
    applySobel(imgNew, filename, "_gauss_5x5_sigma1");
    applyLaplacian(imgNew, filename, "_gauss_5x5_sigma1");
    free(imgNew);

    imgNew = conv(img, 5, kernel_gauss_5x5_sigma2, 0);
    savePGMFile(imgNew, concat(filename, "_gauss_5x5_sigma2.pgm"));
    applySobel(imgNew, filename, "_gauss_5x5_sigma2");
    applyLaplacian(imgNew, filename, "_gauss_5x5_sigma2");
    free(imgNew);

    imgNew = conv(img, 5, kernel_gauss_5x5_sigma4, 0);
    savePGMFile(imgNew, concat(filename, "_gauss_5x5_sigma4.pgm"));
    applySobel(imgNew, filename, "_gauss_5x5_sigma4");
    applyLaplacian(imgNew, filename, "_gauss_5x5_sigma4");
    free(imgNew);

    imgNew = conv(img, 7, kernel_gauss_7x7_sigma1, 0);
    savePGMFile(imgNew, concat(filename, "_gauss_7x7_sigma1.pgm"));
    applySobel(imgNew, filename, "_gauss_7x7_sigma1");
    applyLaplacian(imgNew, filename, "_gauss_7x7_sigma1");
    free(imgNew);

    imgNew = conv(img, 7, kernel_gauss_7x7_sigma2, 0);
    savePGMFile(imgNew, concat(filename, "_gauss_7x7_sigma2.pgm"));
    applySobel(imgNew, filename, "_gauss_7x7_sigma2");
    applyLaplacian(imgNew, filename, "_gauss_7x7_sigma2");
    free(imgNew);

    imgNew = conv(img, 7, kernel_gauss_7x7_sigma4, 0);
    savePGMFile(imgNew, concat(filename, "_gauss_7x7_sigma4.pgm"));
    applySobel(imgNew, filename, "_gauss_7x7_sigma4");
    applyLaplacian(imgNew, filename, "_gauss_7x7_sigma4");
    free(imgNew);

    free(img);

    return 0;
}

/**
 * Applies sobel filter to given image and saves the results
 * */
void applySobel(PGMImage *img, const char *filename, const char *prefix) {
    double kernel_sobelX[MAX_K][MAX_K] = {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1}
    };
    double kernel_sobelY[MAX_K][MAX_K] = {
            {-1, -2, -1},
            {0,  0,  0},
            {1,  2,  1}
    };
    PGMImage *imgSobelX = conv(img, 3, kernel_sobelX, 0);
    PGMImage *imgSobelY = conv(img, 3, kernel_sobelY, 0);
    PGMImage *imgSobel = euclideanDistance(imgSobelX, imgSobelY, 1);
    savePGMFile(imgSobel, concat_three(filename, prefix, "_sobel.pgm"));

    normalize(imgSobelX);
    savePGMFile(imgSobelX, concat_three(filename, prefix, "_sobelX.pgm"));

    normalize(imgSobelY);
    savePGMFile(imgSobelY, concat_three(filename, prefix, "_sobelY.pgm"));

    free(imgSobelX);
    free(imgSobelY);
    free(imgSobel);
}

/**
 * Applies laplacian filter to given image and saves the results
 * */
void applyLaplacian(PGMImage *img, const char *filename, const char *prefix) {
    double kernel_laplacian1[MAX_K][MAX_K] = {
            {0,  -1, 0},
            {-1, 4,  -1},
            {0,  -1, 0}
    };
    double kernel_laplacian2[MAX_K][MAX_K] = {
            {-1, -1, -1},
            {-1, 8,  -1},
            {-1, -1, -1}
    };

    PGMImage *imgLaplacian;
    imgLaplacian = conv(img, 3, kernel_laplacian1, 1);
    savePGMFile(imgLaplacian, concat_three(filename, prefix, "_laplacian1.pgm"));
    free(imgLaplacian);
    
    imgLaplacian = conv(img, 3, kernel_laplacian2, 1);
    savePGMFile(imgLaplacian, concat_three(filename, prefix, "_laplacian2.pgm"));
    free(imgLaplacian);
}

/**
 * Reads ascii (P2) or binary (P5) PGM file
 * and transfers it to given PGMImage struct
 * */
void readPGMFile(char filename[], PGMImage *img) {
    FILE *file;

    // append images/ to filename

    file = fopen(filename, "rb");
    if (file == NULL) {
        printf("!!!ERROR!!! - File cannot be read: %s", filename);
        exit(8);
    }

    printf("Parsing image from file: %s\n", filename);

    if (getc(file) != 'P') {
        printf("!!!ERROR!!! - File is not in PGM file format");
        exit(1);
    }

    int type = getc(file) - '0';

    if (type != 2 && type != 5) {
        printf("!!!ERROR!!! - File is not in either P2 or P5 file format");
        exit(1);
    }

    while (getc(file) != '\n');

    char ch;

    while (getc(file) == '#') {
        printf("#Comment#:");
        while ((ch = getc(file)) != '\n') {
            printf("%c", ch);
        }
        printf("\n");
    }

    fseek(file, -1, SEEK_CUR);

    fscanf(file, "%d %d\n%d", &img->width, &img->height, &img->maxValue);

    if (img->width > MAX_RES) {
        printf("!!!ERROR!!! - width cannot be greater than %d", MAX_RES);
        exit(1);
    }

    if (img->height > MAX_RES) {
        printf("!!!ERROR!!! - height cannot be greater than %d", MAX_RES);
        exit(1);
    }

    printf("width  = %d, height = %d, the highest color value = %d\n", img->width, img->height, img->maxValue);

    int row, col;
    int chInt;

    switch (type) {
        case 2:
            // ascii okuma
            for (row = img->height - 1; row >= 0; row--) {
                for (col = 0; col < img->width; col++) {
                    fscanf(file, "%d", &chInt);
                    img->data[row][col] = chInt;
                }
            }
            break;
        case 5:
            // binary okuma
            while (getc(file) != '\n');

            for (row = img->height - 1; row >= 0; row--) {
                for (col = 0; col < img->width; col++) {
                    img->data[row][col] = (int) ((unsigned char) getc(file));
                }
            }
            break;
    }

    fclose(file);
    printf("File read successfully\n");
}

/**
 * Saves given PGMImage struct as binary (P5) PGM file
 * */
void savePGMFile(PGMImage *img, char filename[]) {
    int rowCount = img->height;
    int colCount = img->width;

    FILE *file = fopen(filename, "wb");

    fprintf(file, "P5\n");
    fprintf(file, "# Created by Taha Korkem (c)2022\n");
    fprintf(file, "%d %d\n%d\n", colCount, rowCount, img->maxValue);

    int i, j;
    for (i = rowCount - 1; i >= 0; i--) {
        for (j = 0; j < colCount; j++) {
            putc(img->data[i][j], file);
        }
    }

    printf("File saved successfully: %s\n", filename);

    fclose(file);
}


void normalize_withMinMax(PGMImage *img, int min, int max) {
    int rowCount = img->height;
    int colCount = img->width;
    int maxVal = img->maxValue;

    int i, j;

    //printf("Normalizasyon yapiliyor: min = %d, max = %d\n", min, max);
    for (i = 0; i < rowCount; i++) {
        for (j = 0; j < colCount; j++) {
            img->data[i][j] = (int) (((float) (img->data[i][j] - min)) / (float) ((max - min)) *
                                     ((float) maxVal));
        }
    }
}

void normalize(PGMImage *img) {
    int rowCount = img->height;
    int colCount = img->width;

    int i, j;

    int min = INT_MAX, max = INT_MIN;

    for (i = 0; i < rowCount; i++) {
        for (j = 0; j < colCount; j++) {
            if (img->data[i][j] < min) {
                min = img->data[i][j];
            }
            if (img->data[i][j] > max) {
                max = img->data[i][j];
            }
        }
    }

    normalize_withMinMax(img, min, max);
}

/**
 * Applies proper convolution to given kernel
 * and returns new PGMImage struct
 * */
PGMImage *conv(PGMImage *img, int n, double kernel[MAX_K][MAX_K], int isNormalize) {
    int rowCount = img->height;
    int colCount = img->width;

    PGMImage *imgNew = calloc(1, sizeof(PGMImage));

    imgNew->width = colCount;
    imgNew->height = rowCount;
    imgNew->maxValue = img->maxValue;

    int minSum = INT_MAX, maxSum = INT_MIN;

    int i, j, k, w;
    for (i = n / 2; i < rowCount - n / 2; i++) {
        for (j = n / 2; j < colCount - n / 2; j++) {

            double sum = 0;

            for (k = 0; k < n; k++) {
                for (w = 0; w < n; w++) {
                    int px = img->data[i + k - n / 2][j + w - n / 2];
                    sum += (px * kernel[k][w]);
                }
            }

            int sumInt = (int) sum;

            if (sumInt > maxSum) {
                maxSum = sumInt;
            }
            if (sumInt < minSum) {
                minSum = sumInt;
            }

            imgNew->data[i][j] = sumInt;
        }
    }

    // normalization
    // (px - min) / (max - min) * 255
    if (isNormalize) {
        normalize_withMinMax(imgNew, minSum, maxSum);
    }

    return imgNew;
}

PGMImage *euclideanDistance(PGMImage *img1, PGMImage *img2, int isNormalize) {
    int rowCount = img1->height;
    int colCount = img1->width;

    PGMImage *imgNew = calloc(1, sizeof(PGMImage));

    imgNew->width = colCount;
    imgNew->height = rowCount;
    imgNew->maxValue = img1->maxValue;

    int minRes = INT_MAX, maxRes = INT_MIN;

    int i, j;
    for (i = 0; i < rowCount; i++) {
        for (j = 0; j < colCount; j++) {

            int px1 = img1->data[i][j];
            int px2 = img2->data[i][j];

            int res = (int) sqrt(pow(px1, 2) + pow(px2, 2));

            if (res > maxRes) {
                maxRes = res;
            }
            if (res < minRes) {
                minRes = res;
            }

            imgNew->data[i][j] = res;
        }
    }

    // normalization
    if (isNormalize) {
        normalize_withMinMax(imgNew, minRes, maxRes);
    }

    return imgNew;
}

char *concat(const char *s1, const char *s2) {
    char *result = malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

char *concat_three(const char *s1, const char *s2, const char *s3) {
    char *result = malloc(strlen(s1) + strlen(s2) + strlen(s3) + 1);
    strcpy(result, s1);
    strcat(result, s2);
    strcat(result, s3);
    return result;
}