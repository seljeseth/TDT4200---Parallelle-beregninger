#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <morph.h>
#include <mpi.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

#include <sys/time.h>
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

double CLAMP(double value, double low, double high)
{
    return (value < low) ? low : ((value > high) ? high : value);
}

#define true 1
#define false 0

//--------------------------------------------------------------------------
//------------------------imgRead-------------------------------------------
//--------------------------------------------------------------------------
void imgRead(const char *filename, pixel **map, int *imgW, int *imgH)
{
    stbi_set_flip_vertically_on_load(true);

    unsigned char *pixelMap;
    int x, y, componentsPerPixel;
    if (strlen(filename) > 0)
    {
        *map = (pixel *)stbi_load(filename, &x, &y, &componentsPerPixel, STBI_rgb_alpha);
    }
    else
    {
        printf("The input file name cannot be empty\n");
        exit(1);
    }

    // Get the current image columns and rows.
    *imgW = x;
    *imgH = y;

    printf("Read the image file %s successfully\n", filename);
}

//--------------------------------------------------------------------------
//------------------------imgWrite-------------------------------------------
//--------------------------------------------------------------------------
void imgWrite(const char *filename, pixel *map, int imgW, int imgH)
{
    if (strlen(filename) < 1)
    {
        printf("The output file name cannot be empty\n");
        exit(1);
    }

    stbi_flip_vertically_on_write(true);
    stbi_write_png(filename, imgW, imgH, STBI_rgb_alpha, map, sizeof(pixel) * imgW);

    printf("The image was written into %s successfully\n", filename);
}

//--------------------------------------------------------------------------------------------------
//--------------------------line interpolating function---------------------------------------------
//--------------------------------------------------------------------------------------------------
void simpleLineInterpolate(const SimpleFeatureLine *sourceLines,
                           const SimpleFeatureLine *destLines,
                           SimpleFeatureLine **morphLines, int numLines,
                           float t)
{
    SimpleFeatureLine *interLines =
        malloc(sizeof(SimpleFeatureLine) * (numLines));

    for (int i = 0; i < numLines; i++)
    {
        interLines[i].startPoint.x = (1 - t) * (sourceLines[i].startPoint.x) +
                                     t * (destLines[i].startPoint.x);
        interLines[i].startPoint.y = (1 - t) * (sourceLines[i].startPoint.y) +
                                     t * (destLines[i].startPoint.y);
        interLines[i].endPoint.x =
            (1 - t) * (sourceLines[i].endPoint.x) + t * (destLines[i].endPoint.x);
        interLines[i].endPoint.y =
            (1 - t) * (sourceLines[i].endPoint.y) + t * (destLines[i].endPoint.y);
    }

    *morphLines = interLines;
}

SimpleFeatureLine **loadLines(int *numLines, const char *name)
{
    FILE *f = fopen(name, "r");
    if (f == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }
    fscanf(f, "%d", numLines);

    char c;
    c = getc(f);

    size_t lineArraySize = sizeof(SimpleFeatureLine) * (*numLines);

    SimpleFeatureLine *linesSrc = (SimpleFeatureLine *)malloc(lineArraySize);

    SimpleFeatureLine *linesDst = (SimpleFeatureLine *)malloc(lineArraySize);

    SimpleFeatureLine **pairs =
        (SimpleFeatureLine **)malloc(sizeof(SimpleFeatureLine *) * 2);

    pairs[0] = linesSrc;
    pairs[1] = linesDst;

    for (int i = 0; i < 2 * (*numLines); i++)
    {
        SimpleFeatureLine *which;
        which = pairs[i % 2];

        int idx = i / 2;

        fscanf(f, "%lf,%lf,%lf,%lf[^\n]", &(which[idx].startPoint.x),
               &(which[idx].startPoint.y), &(which[idx].endPoint.x),
               &(which[idx].endPoint.y));

        c = getc(f);
    }
    return pairs;
}

//---------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------
//--------------------------warp function-----------------------------------------------------------
//--------------------------------------------------------------------------------------------------
/* warping function (backward mapping)
input:
interPt = the point in the intermediary image
interLines = given line in the intermediary image
srcLines = given line in the source image
p, a, b = parameters of the weight function
output:
src = the corresponding point */
void warp(const SimplePoint *interPt, const SimpleFeatureLine *interLines,
          const SimpleFeatureLine *sourceLines, const int sourceLinesSize,
          float p, float a, float b, SimplePoint *src)
{
    int i;
    float interLength, srcLength;
    float weight, weightSum, dist;
    float sum_x, sum_y; // weighted sum of the coordination of the point "src"
    float u, v;
    SimplePoint pd, pq, qd;
    float X, Y;

    sum_x = 0;
    sum_y = 0;
    weightSum = 0;

    for (i = 0; i < sourceLinesSize; i++)
    {
        pd.x = interPt->x - interLines[i].startPoint.x;
        pd.y = interPt->y - interLines[i].startPoint.y;
        pq.x = interLines[i].endPoint.x - interLines[i].startPoint.x;
        pq.y = interLines[i].endPoint.y - interLines[i].startPoint.y;
        interLength = pq.x * pq.x + pq.y * pq.y;
        u = (pd.x * pq.x + pd.y * pq.y) / interLength;

        interLength = sqrt(interLength); // length of the vector PQ

        v = (pd.x * pq.y - pd.y * pq.x) / interLength;

        pq.x = sourceLines[i].endPoint.x - sourceLines[i].startPoint.x;
        pq.y = sourceLines[i].endPoint.y - sourceLines[i].startPoint.y;

        srcLength = sqrt(pq.x * pq.x + pq.y * pq.y); // length of the vector P'Q'
        // corresponding point based on the ith line
        X = sourceLines[i].startPoint.x + u * pq.x + v * pq.y / srcLength;
        Y = sourceLines[i].startPoint.y + u * pq.y - v * pq.x / srcLength;

        // the distance from the corresponding point to the line P'Q'
        if (u < 0)
            dist = sqrt(pd.x * pd.x + pd.y * pd.y);
        else if (u > 1)
        {
            qd.x = interPt->x - interLines[i].endPoint.x;
            qd.y = interPt->y - interLines[i].endPoint.y;
            dist = sqrt(qd.x * qd.x + qd.y * qd.y);
        }
        else
        {
            dist = fabsf(v);
        }

        weight = pow(pow(interLength, p) / (a + dist), b);
        sum_x += X * weight;
        sum_y += Y * weight;
        weightSum += weight;
    }

    src->x = sum_x / weightSum;
    src->y = sum_y / weightSum;
}

//--------------------------------------------------------------------------------------------------
//--------------------------bilinear interpolation--------------------------------------------------
//--------------------------------------------------------------------------------------------------
void bilinear(pixel *Im, float row, float col, pixel *pix)
{
    int cm, cn, fm, fn;
    double alpha, beta;

    cm = (int)ceil(row);
    fm = (int)floor(row);
    cn = (int)ceil(col);
    fn = (int)floor(col);
    alpha = ceil(row) - row;
    beta = ceil(col) - col;

    pix->r = (unsigned int)(alpha * beta * Im[fm * imgWidthOrig + fn].r + (1 - alpha) * beta * Im[cm * imgWidthOrig + fn].r + alpha * (1 - beta) * Im[fm * imgWidthOrig + cn].r + (1 - alpha) * (1 - beta) * Im[cm * imgWidthOrig + cn].r);
    pix->g = (unsigned int)(alpha * beta * Im[fm * imgWidthOrig + fn].g + (1 - alpha) * beta * Im[cm * imgWidthOrig + fn].g + alpha * (1 - beta) * Im[fm * imgWidthOrig + cn].g + (1 - alpha) * (1 - beta) * Im[cm * imgWidthOrig + cn].g);
    pix->b = (unsigned int)(alpha * beta * Im[fm * imgWidthOrig + fn].b + (1 - alpha) * beta * Im[cm * imgWidthOrig + fn].b + alpha * (1 - beta) * Im[fm * imgWidthOrig + cn].b + (1 - alpha) * (1 - beta) * Im[cm * imgWidthOrig + cn].b);
    pix->a = 255;
}
//---------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------
//--------------------------color interpolation function--------------------------------------------
//--------------------------------------------------------------------------------------------------
void ColorInterPolate(const SimplePoint *Src_P,
                      const SimplePoint *Dest_P, float t,
                      pixel *imgSrc, pixel *imgDest, pixel *rgb)
{
    pixel srcColor, destColor;

    bilinear(imgSrc, Src_P->y, Src_P->x, &srcColor);
    bilinear(imgDest, Dest_P->y, Dest_P->x, &destColor);

    rgb->b = srcColor.b * (1 - t) + destColor.b * t;
    rgb->g = srcColor.g * (1 - t) + destColor.g * t;
    rgb->r = srcColor.r * (1 - t) + destColor.r * t;
    rgb->a = 255;
}
//---------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------
//--------------------------morph-------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------

void morphKernel(const SimpleFeatureLine *hSrcLines,
                 const SimpleFeatureLine *hDstLines,
                 SimpleFeatureLine *hMorphLines,
                 pixel *hSrcImgMap,
                 pixel *hDstImgMap,
                 pixel *hMorphMap,
                 int numLines,
                 float t)
{
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int local_height = imgHeightOrig / size;
    int offset = local_height * rank;

    //each process only needs to go through its own pixels which will be in its local height
    for (int i = 0; i < local_height; i++)
    {
        for (int j = 0; j < imgWidthOrig; j++)
        {
            pixel interColor;
            SimplePoint dest;
            SimplePoint src;
            SimplePoint q;
            q.x = j;
            //makes sure that each process starts at different locations which is given by offset
            q.y = i + offset;

            // warping
            warp(&q, hMorphLines, hSrcLines, numLines, p, a, b, &src);
            warp(&q, hMorphLines, hDstLines, numLines, p, a, b, &dest);

            src.x = CLAMP(src.x, 0, imgWidthOrig - 1);
            src.y = CLAMP(src.y, 0, imgHeightOrig - 1);
            dest.x = CLAMP(dest.x, 0, imgWidthOrig - 1);
            dest.y = CLAMP(dest.y, 0, imgHeightOrig - 1);

            // color interpolation
            ColorInterPolate(&src, &dest, t, hSrcImgMap, hDstImgMap, &interColor);

            hMorphMap[i * imgWidthOrig + j].r = interColor.r;
            hMorphMap[i * imgWidthOrig + j].g = interColor.g;
            hMorphMap[i * imgWidthOrig + j].b = interColor.b;
            hMorphMap[i * imgWidthOrig + j].a = interColor.a;
        }
    }
}

void doMorph(const SimpleFeatureLine *hSrcLines, const SimpleFeatureLine *hDstLines, int numLines, float t)
{

    int world_size;
    int world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    ///////////////////////////////
    // CREATE INTERPOLATED LINES //
    ///////////////////////////////
    SimpleFeatureLine *hMorphLines = NULL;
    simpleLineInterpolate(hSrcLines, hDstLines, &hMorphLines, numLines, t);

    ////////////////////////////////
    // PERFORM THE MORPHING STAGE //
    ////////////////////////////////
    struct timeval start, end;
    gettimeofday(&start, NULL);
    morphKernel(hSrcLines,
                hDstLines,
                hMorphLines,
                hSrcImgMap,
                hDstImgMap,
                hMorphMap,
                numLines,
                t);
    gettimeofday(&end, NULL);
    printf("Morph time: %.2f seconds\n", WALLTIME(end) - WALLTIME(start));

    //////////////////////////////////
    // WRITE OUT THE FINISHED IMAGE //
    //////////////////////////////////

    // Getting all of the image slices and merge them
    MPI_Gather(hMorphMap, (sizeof(pixel) * imgWidthOrig * (imgHeightOrig / world_size)), MPI_BYTE, hMorphMap, (sizeof(pixel) * imgWidthOrig * (imgHeightOrig / world_size)), MPI_BYTE, 0, MPI_COMM_WORLD);
    // Rank 0 writes image to file
    if (world_rank == 0)
    {
        char rootFile[50] = {0};
        sprintf(rootFile, "%s%.5f.png", outputFile, t);
        imgWrite(rootFile, hMorphMap, imgWidthOrig, imgHeightOrig);
    }

    free(hMorphLines);
}

//------------main function----------------------------
int main(int argc, char *argv[])
{

    //////////////////////////////////
    // MPI INITIALIZATION AND SETUP //
    //////////////////////////////////
    MPI_Init(&argc, &argv);
    int world_size, world_rank;

    //Getting world size and world rank
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    /////////////////////////////////////
    // ARGUMENT PARSING - DO NOT TOUCH //
    /////////////////////////////////////
    int steps;
    const char *stepsStr;

    if (world_rank == 0)
    {
        switch (argc)
        {
        case 6:
            inputFileOrig = argv[1];
            inputFileDest = argv[2];
            outputFile = argv[3];
            stepsStr = argv[4];
            linePath = argv[5];

            printf("Input File Source: %s\n", inputFileOrig);
            printf("Input File Dest: %s\n", inputFileDest);
            steps = atoi(stepsStr);
            imgRead(inputFileOrig, &hSrcImgMap, &imgWidthOrig, &imgHeightOrig);
            imgRead(inputFileDest, &hDstImgMap, &imgWidthDest, &imgHeightDest);
            break;

        case 9:
            inputFileOrig = argv[1];
            inputFileDest = argv[2];
            outputFile = argv[3];
            stepsStr = argv[4];
            linePath = argv[5];
            pStr = argv[6];
            aStr = argv[7];
            bStr = argv[8];

            steps = atoi(stepsStr);
            p = atof(pStr);
            a = atof(aStr);
            b = atof(bStr);
            imgRead(inputFileOrig, &hSrcImgMap, &imgWidthOrig, &imgHeightOrig);
            imgRead(inputFileDest, &hDstImgMap, &imgWidthDest, &imgHeightDest);
            break;

        default:
            printf("Usage\n");
            printf("./morph sourceImage.png destinationImage.png outputpath steps linePath [p] [a] [b]\n");
            exit(1);
        }
    }
    /////////////////////////////
    // END OF ARGUMENT PARSING //
    /////////////////////////////

    // Broadcasting arguments
    MPI_Bcast(&p, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&a, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&b, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    //Broadcasting the remaining arguments
    MPI_Bcast(&imgWidthOrig, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&imgHeightOrig, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&steps, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /////////////////////////////
    // Load the feature lines  //
    /////////////////////////////
    int numLines;
    SimpleFeatureLine *hSrcLines;
    SimpleFeatureLine *hDstLines;

    if (world_rank == 0)
    {
        SimpleFeatureLine **hLinePairs = loadLines(&numLines, linePath);
        hSrcLines = hLinePairs[0];
        hDstLines = hLinePairs[1];

        free(hLinePairs);
    }

    // Broadcasting the number of lines
    MPI_Bcast(&numLines, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Since we already have the values for rank 0 we only need to allocate to the other processes
    if (world_rank != 0)
    {
        hSrcImgMap = (pixel *)malloc(sizeof(pixel) * imgWidthOrig * imgHeightOrig);
        if (hSrcImgMap == NULL)
        {
            exit(1);
        }
        hDstImgMap = (pixel *)malloc(sizeof(pixel) * imgWidthOrig * imgHeightOrig);
        if (hDstImgMap == NULL)
        {
            exit(1);
        }
        hSrcLines = (SimpleFeatureLine *)malloc(sizeof(SimpleFeatureLine) * numLines);
        if (hSrcLines == NULL)
        {
            exit(1);
        }
        hDstLines = (SimpleFeatureLine *)malloc(sizeof(SimpleFeatureLine) * numLines);
        if (hDstLines == NULL)
        {
            exit(1);
        }
    }

    // We divide the image for each process into equal sized images, the height of that image is our local_height
    int local_height = imgHeightOrig / world_size;

    // Broadcasting linepairs
    MPI_Bcast(hSrcLines, sizeof(SimpleFeatureLine) * numLines, MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(hDstLines, sizeof(SimpleFeatureLine) * numLines, MPI_BYTE, 0, MPI_COMM_WORLD);

    //  Broadcasting both input images
    MPI_Bcast(hSrcImgMap, (sizeof(pixel) * imgWidthOrig * imgHeightOrig), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(hDstImgMap, (sizeof(pixel) * imgWidthOrig * imgHeightOrig), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Run steps with size t
    // Allocating enough space for the whole image to process one, since it will be the one writing the image
    // for the other ranks we only need enough for the local partition of the image
    if (world_rank == 0)
    {
        hMorphMap = (pixel *)malloc(sizeof(pixel) * imgWidthOrig * imgHeightOrig);
        if (hMorphMap == NULL)
        {
            exit(1);
        }
    }
    else
    {
        hMorphMap = (pixel *)malloc(sizeof(pixel) * imgWidthOrig * local_height);
        if (hMorphMap == NULL)
        {
            exit(1);
        }
    }
    float stepSize = 1.0 / steps;
    for (int i = 0; i < steps + 1; i++)
    {
        t = stepSize * i;
        doMorph(hSrcLines, hDstLines, numLines, t);
    }

    free(hSrcLines);
    free(hDstLines);
    free(hSrcImgMap);
    free(hDstImgMap);
    free(hMorphMap); //since the local image buffers share the same variable we have to free it for everyone, not just rank 0.
    MPI_Finalize();
    return 0;
}
