#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <chrono>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/vec3.hpp> // vec3
#include <glm/vec4.hpp> // vec4
#include <glm/mat4x4.hpp>
#include "stb_image.h"
#include <SOIL2/SOIL2.h>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>

#pragma comment(lib, "glfw3.lib")
#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "opengl32.lib")

using namespace std;
using namespace glm;

mat4 ModelObj = mat4(1.0f);
//размерность n x n
GLint n_size = 10;
//Точка отсчета
GLfloat x0 = -(GLfloat)n_size / 2.0; //x
GLfloat z0 = x0;//z
//Шаг
GLfloat dx = 0.01; //по x
GLfloat dz = dx; //по z
//Количество ячеек
GLint Tess_N = n_size / dx;

GLFWwindow* g_window;

GLuint g_shaderProgram;
GLint g_uMVP;
GLint g_uMV;
GLint g_mix_ratio;
GLfloat mix_ratio = 0;
const int countTextures = 2;
GLuint texIDs[countTextures];

struct Texture
{
    const GLchar* fileName;
    const GLchar* uniformMapName;
    GLint mapLocation;
    GLint texUnit;
    const GLchar* format;

    Texture(const GLchar* fileName, const GLchar* uniformMapName, GLint texUnit, const GLchar* format)
    {
        this->format = format;
        this->fileName = fileName;
        this->uniformMapName = uniformMapName;
        this->texUnit = texUnit;
    }
};

Texture textures[countTextures] =
{
    Texture("C:\\Users\\margo\\source\\repos\\OpenGLDemo2\\x64\\Debug\\first_texture.png", "u_map1", 0, "png"),
    Texture("C:\\Users\\margo\\source\\repos\\OpenGLDemo2\\x64\\Debug\\second_texture.jpg", "u_map2", 1, "jpg")
};




float func(float x, float z) {
    return 0.25 * (1 - x * z) * sinf(1 - x * z);
}

float d_f_x(float x, float z) {
    return -0.25 * z * sinf(1 - x * z) - 0.25 * z * (1 - x * z) * cosf(-1 + x * z);
}

float d_f_z(float x, float z) {
    return -0.25 * x * sinf(1 - x * z) - 0.25 * x * (1 - x * z) * cosf(-1 + x * z);
}


chrono::time_point<chrono::system_clock> g_callTime;
class MyMatrix {
    float m_mat[16];
public:
    void translation(float x, float y, float z) { //матрица переноса

        m_mat[0] = 1.0f; m_mat[1] = 0.0f; m_mat[2] = 0.0f; m_mat[3] = 0.0f;
        m_mat[4] = 0.0f; m_mat[5] = 1.0f; m_mat[6] = 0.0f; m_mat[7] = 0.0f;
        m_mat[8] = 0.0f; m_mat[9] = 0.0f; m_mat[10] = 1.0f; m_mat[11] = 0.0f;
        m_mat[12] = x; m_mat[13] = y; m_mat[14] = z; m_mat[15] = 1.0f;

    }
    void rotate(float angle, float x, float y, float z) {
        // Вычисляем тригонометрические функции угла поворота
        float c = cos(angle);
        float s = sin(angle);
        float t = 1.0f - c;
        float len = sqrt(x * x + y * y + z * z);

        // Нормализуем вектор оси поворота
        if (len != 0.0f) {
            x /= len;
            y /= len;
            z /= len;
        }

        // Заполняем матрицу поворота
        m_mat[0] = c + x * x * t;
        m_mat[1] = y * x * t + z * s;
        m_mat[2] = z * x * t - y * s;
        m_mat[3] = 0.0f;

        m_mat[4] = x * y * t - z * s;
        m_mat[5] = c + y * y * t;
        m_mat[6] = z * y * t + x * s;
        m_mat[7] = 0.0f;

        m_mat[8] = x * z * t + y * s;
        m_mat[9] = y * z * t - x * s;
        m_mat[10] = c + z * z * t;
        m_mat[11] = 0.0f;

        m_mat[12] = 0.0f;
        m_mat[13] = 0.0f;
        m_mat[14] = 0.0f;
        m_mat[15] = 1.0f;
    }

    MyMatrix operator * (const MyMatrix& m) {
        MyMatrix mat;
        const float* a = m_mat;
        const float* b = m.m_mat;
        float* result = mat.m_mat;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                result[i * 4 + j] =
                    a[i * 4] * b[j] +
                    a[i * 4 + 1] * b[j + 4] +
                    a[i * 4 + 2] * b[j + 8] +
                    a[i * 4 + 3] * b[j + 12];
            }
        }
        return mat;
    }
    friend MyMatrix operator*(const mat4& left, const MyMatrix& right) {
        MyMatrix mat;
        const float* b = right.m_mat;
        float a[16]; int k = 0;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                a[k] = left[j][i]; k++;
            }
            std::cout << std::endl;
        }
        //const float* b = value_ptr(m);
        float* result = mat.m_mat;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                result[i * 4 + j] =
                    a[i * 4] * b[j] +
                    a[i * 4 + 1] * b[j + 4] +
                    a[i * 4 + 2] * b[j + 8] +
                    a[i * 4 + 3] * b[j + 12];
            }
        }
        return mat;
    }
    MyMatrix operator * (const mat4& m) {
        MyMatrix mat;
        const float* a = m_mat;
        float b[16]; int k = 0;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                b[k] = m[j][i]; k++;
            }
            std::cout << std::endl;
        }
        //const float* b = value_ptr(m);
        float* result = mat.m_mat;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                result[i * 4 + j] =
                    a[i * 4] * b[j] +
                    a[i * 4 + 1] * b[j + 4] +
                    a[i * 4 + 2] * b[j + 8] +
                    a[i * 4 + 3] * b[j + 12];
            }
        }
        return mat;
    }
    void  scale(float x, float y, float z) {

        m_mat[0] = x;
        m_mat[1] = 0.0f;
        m_mat[2] = 0.0f;
        m_mat[3] = 0.0f;
        m_mat[4] = 0.0f;
        m_mat[5] = y;
        m_mat[6] = 0.0f;
        m_mat[7] = 0.0f;
        m_mat[8] = 0.0f;
        m_mat[9] = 0.0f;
        m_mat[10] = z;
        m_mat[11] = 0.0f;
        m_mat[12] = 0.0f;
        m_mat[13] = 0.0f;
        m_mat[14] = 0.0f;
        m_mat[15] = 1.0f;

    }

    void Projection(float n, float r, float t, float l, float b, float f) {

        m_mat[0] = 2 * n / (r - l);
        m_mat[1] = 0.0f;
        m_mat[2] = (r + l) / (r - l);
        m_mat[3] = 0.0f;
        m_mat[4] = 0.0f;
        m_mat[5] = 2 * n / (t - b);
        m_mat[6] = (t + b) / (t - b);
        m_mat[7] = 0.0f;
        m_mat[8] = 0.0f;
        m_mat[9] = 0.0f;
        m_mat[10] = -(f + n) / (f - n);
        m_mat[11] = -2 * f * n / (f - n);
        m_mat[12] = 0.0f;
        m_mat[13] = 0.0f;
        m_mat[14] = -1.0f;
        m_mat[15] = 0.0f;

    }
    void View() {
        m_mat[0] =
            0.514496;
        m_mat[1] = 0.0;
        m_mat[2] = -0.857493;
        m_mat[3] = 9.53674e-07;
        m_mat[4] = -0.392299;
        m_mat[5] = 0.889212;
        m_mat[6] = -0.23538;
        m_mat[7] = -4.76837e-07;
        m_mat[8] = 0.762493;
        m_mat[9] = 0.457496;
        m_mat[10] = 0.457496;
        m_mat[11] = -32.7872;
        m_mat[12] = 0.0;
        m_mat[13] = 0.0;
        m_mat[14] = 0.0;
        m_mat[15] = 1.0;

    }
    MyMatrix() {
        m_mat[0] =
            1.0f;
        m_mat[1] = 0.0f;
        m_mat[2] = 0.0f;
        m_mat[3] = 0.0f;
        m_mat[4] = 0.0f;
        m_mat[5] = 1.0f;
        m_mat[6] = 0.0f;
        m_mat[7] = 0.0f;
        m_mat[8] = 0.0f;
        m_mat[9] = 0.0f;
        m_mat[10] = 1.0f;
        m_mat[11] = 0.0f;
        m_mat[12] = 0.0f;
        m_mat[13] = 0.0f;
        m_mat[14] = 0.0f;
        m_mat[15] = 1.0f;
    }
    void Model() {
        m_mat[0] =
            1.0f;
        m_mat[1] = 0.0f;
        m_mat[2] = 0.0f;
        m_mat[3] = 0.0f;
        m_mat[4] = 0.0f;
        m_mat[5] = 1.0f;
        m_mat[6] = 0.0f;
        m_mat[7] = 0.0f;
        m_mat[8] = 0.0f;
        m_mat[9] = 0.0f;
        m_mat[10] = 1.0f;
        m_mat[11] = 0.0f;
        m_mat[12] = 0.0f;
        m_mat[13] = 0.0f;
        m_mat[14] = 0.0f;
        m_mat[15] = 1.0f;
    }

    float* get() { return m_mat; };
};
MyMatrix* M = new MyMatrix();


class Model
{
public:
    GLuint vbo;
    GLuint ibo;
    GLuint vao;
    GLsizei indexCount;
};

Model g_model;

GLuint createShader(const GLchar* code, GLenum type)
{
    GLuint result = glCreateShader(type);

    glShaderSource(result, 1, &code, NULL);
    glCompileShader(result);

    GLint compiled;
    glGetShaderiv(result, GL_COMPILE_STATUS, &compiled);

    if (!compiled)
    {
        GLint infoLen = 0;
        glGetShaderiv(result, GL_INFO_LOG_LENGTH, &infoLen);
        if (infoLen > 0)
        {
            char* infoLog = (char*)alloca(infoLen);
            glGetShaderInfoLog(result, infoLen, NULL, infoLog);
            cout << "Shader compilation error" << endl << infoLog << endl;
        }
        glDeleteShader(result);
        return 0;
    }

    return result;
}

GLuint createProgram(GLuint vsh, GLuint fsh)
{
    GLuint result = glCreateProgram();

    glAttachShader(result, vsh);
    glAttachShader(result, fsh);

    glLinkProgram(result);

    GLint linked;
    glGetProgramiv(result, GL_LINK_STATUS, &linked);

    if (!linked)
    {
        GLint infoLen = 0;
        glGetProgramiv(result, GL_INFO_LOG_LENGTH, &infoLen);
        if (infoLen > 0)
        {
            char* infoLog = (char*)alloca(infoLen);
            glGetProgramInfoLog(result, infoLen, NULL, infoLog);
            cout << "Shader program linking error" << endl << infoLog << endl;
        }
        glDeleteProgram(result);
        return 0;
    }

    return result;
}
void generateTexture(const GLchar* file_name, GLint texID)
{
    GLint texW, texH;
    GLint channels;
    unsigned char* image = SOIL_load_image(file_name, &texW, &texH, &channels, SOIL_LOAD_RGBA);
    cout << SOIL_last_result() << endl;
    cout << channels << endl;

    glBindTexture(GL_TEXTURE_2D, texID);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texW, texH, 0, GL_RGBA, GL_UNSIGNED_BYTE, image);
    SOIL_free_image_data(image);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);



    //glEnable(GL_GENERATE_MIPMAP);
    //glGenerateMipmap(GL_TEXTURE_2D);


    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);


    //glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_LOD_BIAS, -0.4f);
    //glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_LOD, -1.0f);
    //glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_LOD, 0.0f);




    //// анизотропная
    //glEnable(GL_TEXTURE_MAX_ANISOTROPY_EXT);
    //glEnable(GL_GENERATE_MIPMAP);
    //glGenerateMipmap(GL_TEXTURE_2D);
    //float maxAnisotropy;
    //glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &maxAnisotropy);

    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    //glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, maxAnisotropy);
    //glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_LOD_BIAS, 1.0f);


}

void initTextures()
{

    glGenTextures(2, texIDs);

    for (int i = 0; i < countTextures; i++)
    {
        generateTexture(textures[i].fileName, texIDs[i]);
        textures[i].mapLocation = glGetUniformLocation(g_shaderProgram, textures[i].uniformMapName);
    }
}
bool createShaderProgram() {
    g_shaderProgram = 0;

    const GLchar vsh[] =
        "#version 330\n"
        ""
        "layout(location = 0) in vec3 a_position;"
        "layout(location = 1) in vec3 a_normal; "
        "layout(location = 2) in vec2 a_texCoord;"
        ""
        "uniform mat4 u_mv;"
        "uniform mat4 u_mvp;"

        ""
        "out vec3 v_p;"
        "out vec3 v_normal;"
        "out vec2 v_texCoord;"
        ""
        "void main()"
        "{"
        "    float x = a_position[0];"
        "    float z = a_position[2];"
        "    vec4 pos = vec4(a_position[0], a_position[1], a_position[2], 1.0);"
        "    gl_Position = u_mvp*pos;"
        "    v_p = (u_mv*pos).xyz;"
        "    v_normal = normalize(transpose(inverse(mat3(u_mv)))*a_normal);"
        "    v_texCoord = a_texCoord;"
        "}"
        ;

    const GLchar fsh[] =
        "#version 330\n"
        ""
        "in vec3 v_p;"
        "in vec3 v_normal;"
        "in vec2 v_texCoord;"
        ""
        "layout(location = 0) out vec4 o_color;"
        "uniform sampler2D u_map1;"
        "uniform sampler2D u_map2;"
        "uniform float mix_ratio;"
        ""
        "void main()"
        "{"
        "   vec3 u_l = vec3(5.0, 10.0, 0.0);"
        "   vec3 n = normalize(v_normal);"
        "   vec3 l = normalize(v_p-u_l);"
        "   float a = dot(-l, n);"
        "   float d = max(a, 0.1);"
        "   float n2 = 15.;"
        "   float d2 = 60.;"

        "   vec3 e = normalize(-v_p);"
        "   vec3 h = normalize(-l+ e);"
        "   float s = pow(max(dot(h, n), 0.0), 5.0);" //компонент зеркального блика

        "vec4 texel = mix(texture(u_map1, v_texCoord), texture(u_map2, v_texCoord), mix_ratio);"

        "   o_color = vec4(d*texel.xyz + s*vec3(1.0), 1.0);"
        //sin((sin(v_texCoord.x * n2) / n2 - v_texCoord.y) * d2
        "}"
        ;


    GLuint vertexShader, fragmentShader;

    vertexShader = createShader(vsh, GL_VERTEX_SHADER);
    fragmentShader = createShader(fsh, GL_FRAGMENT_SHADER);

    g_shaderProgram = createProgram(vertexShader, fragmentShader);

    //Матрицы
    g_uMV = glGetUniformLocation(g_shaderProgram, "u_mv");
    g_uMVP = glGetUniformLocation(g_shaderProgram, "u_mvp");
    g_mix_ratio = glGetUniformLocation(g_shaderProgram, "mix_ratio");

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    initTextures();

    return g_shaderProgram != 0;
}

//Получение порядкового номера вершины по индексам i, j
GLint getNumVertex(int j, int i, int n) {
    return (GLint)(i + j * (n + 1));
}
void generateMesh(GLfloat* vertices, GLuint* indices, int Tess_N) {
    int index = 0;

    GLfloat* arr_x = new GLfloat[Tess_N + 1];//координаты x
    GLfloat* arr_z = new GLfloat[Tess_N + 1];//координаты z

    for (int i = 0; i <= Tess_N; i++) {
        arr_x[i] = x0 + i * dx;
    }


    for (int i = 0; i <= Tess_N; i++) {
        arr_z[i] = x0 + i * dz;
    }


    int k = 0;
    for (int j = 0; j <= Tess_N; j++) {
        for (int i = 0; i <= Tess_N; i++) {

            vertices[k] = arr_x[i];
            k++;
            vertices[k] = func(arr_x[i], arr_z[j]);
            k++;
            vertices[k] = arr_z[j];
            k++;
            vertices[k] = d_f_x(arr_x[i], arr_z[j]);
            k++;
            vertices[k] = 1.0;
            k++;
            vertices[k] = d_f_z(arr_x[i], arr_z[j]);
            k++;
            vertices[k] = (arr_x[i] + 5.0f) / 10.0f;
            k++;
            vertices[k] = (arr_z[j] + 5.0f) / 10.0f;
            k++;
        }
    }
    //for (int x = 0; x <= Tess_N; x++) {
    //  for (int y = 0; y <= Tess_N; y++) {
    //      float f_x = x / (GLfloat)Tess_N;
    //      float f_y = y / (GLfloat)Tess_N;

    //      vertices[(x * Tess_N + y) * 2 + 0] = f_x;
    //      
    //      vertices[(x * Tess_N + y) * 2 + 1] = f_y;

    //      
    //  }
    //}


     //Заполняем массив индексов
    k = 0;
    int j = 0;
    while (j < Tess_N) {
        for (int i = 0; i <= Tess_N; i++) {
            indices[k] = getNumVertex(j, i, Tess_N);
            k++;
            indices[k] = getNumVertex(j + 1, i, Tess_N);
            k++;
        }
        if (j < Tess_N - 1) {
            indices[k] = getNumVertex(j + 1, Tess_N, Tess_N);
            k++;
        }
        j++;
        if (j < Tess_N) {
            for (int i = Tess_N; i >= 0; i--) {
                indices[k] = getNumVertex(j, i, Tess_N);
                k++;
                indices[k] = getNumVertex(j + 1, i, Tess_N);
                k++;
            }
            if (j < Tess_N - 1) {
                indices[k] = getNumVertex(j + 1, 0, Tess_N);
                k++;
            }
            j++;
        }
    }
    /*for (int x = 0; x < Tess_N; x++) {
        for (int y = 0; y < Tess_N ; y++) {
            indices[(x * (Tess_N - 1) + y) * 2 * 3 + 0] = (x + 0) * Tess_N + y + 0;
            indices[(x * (Tess_N - 1) + y) * 2 * 3 + 1] = (x + 1) * Tess_N + y + 1;
            indices[(x * (Tess_N - 1) + y) * 2 * 3 + 2] = (x + 1) * Tess_N + y + 0;

            indices[(x * (Tess_N - 1) + y) * 2 * 3 + 3] = (x + 0) * Tess_N + y + 0;
            indices[(x * (Tess_N - 1) + y) * 2 * 3 + 4] = (x + 0) * Tess_N + y + 1;
            indices[(x * (Tess_N - 1) + y) * 2 * 3 + 5] = (x + 1) * Tess_N + y + 1;
        }
    }*/
}
bool createModel() {
    GLint arr_vertex_size = (Tess_N + 1) * (Tess_N + 1) * 8; //Размерность одномерного массива с вершинами
    GLint arr_index_size = 2 * (Tess_N + 1) * Tess_N + (Tess_N - 1); //Размерность одномерного массива с индексами


    GLfloat* vertices = (GLfloat*)malloc(arr_vertex_size * sizeof(GLfloat));;
    GLuint* indices = (GLuint*)malloc(
        arr_index_size * sizeof(GLuint)
    ); //Создали массив с индексами обхода

    /*for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            GLfloat x = (GLfloat)i / n;
            GLfloat z = (GLfloat)j / n;
            vertices[(i * n + j) * 2] = x;
            vertices[(i * n + j) * 2 + 1] = z;
        }
    }
    */
    generateMesh(vertices, indices, Tess_N);

    glGenVertexArrays(1, &g_model.vao);
    glBindVertexArray(g_model.vao);

    glGenBuffers(1, &g_model.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, g_model.vbo);
    glBufferData(GL_ARRAY_BUFFER, arr_vertex_size * sizeof(GLfloat), vertices, GL_STATIC_DRAW);

    glGenBuffers(1, &g_model.ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_model.ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, arr_index_size * sizeof(GLint), indices, GL_STATIC_DRAW);

    g_model.indexCount = arr_index_size;



    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (const GLvoid*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (const GLvoid*)(3 * sizeof(GLfloat)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (const GLvoid*)(6 * sizeof(GLfloat)));

    free(indices);
    free(vertices);

    return g_model.vbo != 0 && g_model.ibo != 0 && g_model.vao != 0;
}

bool init() {
    // Set initial color of color buffer to white.
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);
    glEnable(GL_CULL_FACE);

    return createShaderProgram() && createModel();
}

void reshape(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}
float max(float a, float b) {
    return a > b ? a : b;
}
void draw(GLfloat delta_draw)
{

    // Clear color buffer.
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(g_shaderProgram);
    glBindVertexArray(g_model.vao);


    mat4 Projection = perspective(radians(45.0f), 4.0f / 3.0f, x0, abs(x0));


    mat4 View = lookAt(
        vec3(5, 3, 0) * abs(x0),
        vec3(0, 0, 0),
        vec3(0, 1, 0)
    );

    GLfloat a = delta_draw;


    if (glfwGetKey(g_window, GLFW_KEY_LEFT) || glfwGetKey(g_window, GLFW_KEY_RIGHT)) {
        ModelObj = rotate(ModelObj, delta_draw, vec3(0.0, 1.0, 0.0));
    }
    else if (glfwGetKey(g_window, GLFW_KEY_UP) || glfwGetKey(g_window, GLFW_KEY_DOWN))
    {
        ModelObj = rotate(ModelObj, delta_draw, vec3(0.0, 0.0, 1.0));
    }
    else if (glfwGetKey(g_window, GLFW_KEY_W))
    {
        if (mix_ratio < 1)
            mix_ratio += 0.001;
    }
    else if (glfwGetKey(g_window, GLFW_KEY_S))
    {
        if (mix_ratio > 0)
            mix_ratio -= 0.001;
    }
    //Матрица MV
    mat4 MV = View * ModelObj;

    //Матрица MVP
    mat4 MVP = Projection * MV;



    glUniformMatrix4fv(g_uMV, 1, GL_FALSE, &MV[0][0]);
    glUniformMatrix4fv(g_uMVP, 1, GL_FALSE, &MVP[0][0]);
    glUniform1f(g_mix_ratio, mix_ratio);
    for (int i = 0; i < countTextures; i++)
    {
        glActiveTexture(GL_TEXTURE0 + textures[i].texUnit);
        glBindTexture(GL_TEXTURE_2D, texIDs[i]);
        glUniform1i(textures[i].mapLocation, textures[i].texUnit);
    }
    glDrawElements(GL_TRIANGLE_STRIP, g_model.indexCount, GL_UNSIGNED_INT, NULL);
}

void cleanup()
{
    if (g_shaderProgram != 0)
        glDeleteProgram(g_shaderProgram);
    if (g_model.vbo != 0)
        glDeleteBuffers(1, &g_model.vbo);
    if (g_model.ibo != 0)
        glDeleteBuffers(1, &g_model.ibo);
    if (g_model.vao != 0)
        glDeleteVertexArrays(1, &g_model.vao);
    glDeleteTextures(countTextures, texIDs);
}

bool initOpenGL()
{
    // Initialize GLFW functions.
    if (!glfwInit())
    {
        cout << "Failed to initialize GLFW" << endl;
        return false;
    }

    // Request OpenGL 3.3 without obsoleted functions.
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create window.
    g_window = glfwCreateWindow(800, 600, "OpenGL Test", NULL, NULL);
    if (g_window == NULL)
    {
        cout << "Failed to open GLFW window" << endl;
        glfwTerminate();
        return false;
    }

    // Initialize OpenGL context with.
    glfwMakeContextCurrent(g_window);

    // Set internal GLEW variable to activate OpenGL core profile.
    glewExperimental = true;

    // Initialize GLEW functions.
    if (glewInit() != GLEW_OK)
    {
        cout << "Failed to initialize GLEW" << endl;
        return false;
    }

    // Ensure we can capture the escape key being pressed.
    glfwSetInputMode(g_window, GLFW_STICKY_KEYS, GL_TRUE);

    // Set callback for framebuffer resizing event.
    glfwSetFramebufferSizeCallback(g_window, reshape);

    return true;
}

void tearDownOpenGL()
{
    // Terminate GLFW.
    glfwTerminate();
}

int main()
{
    //cout << Tess_N << ", " << x0 << ", " << dx;
    // Initialize OpenGL
    if (!initOpenGL())
        return -1;
    /* MyMatrix T;
     T.translation(-50.0f, 0, -100.0f);
     MyMatrix S;
     float x = x0;
     float z = z0;
     MyMatrix  R;
     S.scale(1/200.0f, 1 / 200.0f, 1 / 200.0f);

     R.rotate(radians(-45.0), 1, 0, 0);


     (*M) = T * R ;
     R.rotate(radians(45.0), 0, 0, 1);
     (*M) = (*M)*R*S;
     MyMatrix V;
     V.View();
     (*M) = V*(*M);*/
     // Initialize graphical resources.
    bool isOk = init();

    if (isOk)
    {
        GLfloat lastTime = glfwGetTime();

        // Main loop until window closed or escape pressed.
        while (glfwGetKey(g_window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(g_window) == 0)
        {
            GLfloat currentTime = glfwGetTime();
            GLfloat deltaTime = GLfloat(currentTime - lastTime);
            lastTime = currentTime;

            GLfloat delta = 0;
            GLfloat angle = 200.0;
            if (glfwGetKey(g_window, GLFW_KEY_LEFT) || glfwGetKey(g_window, GLFW_KEY_UP)) {
                delta = radians(angle) * deltaTime;
            }
            else if (glfwGetKey(g_window, GLFW_KEY_RIGHT) || glfwGetKey(g_window, GLFW_KEY_DOWN)) {
                delta = -radians(angle) * deltaTime;
            }
            draw(delta);
            // Swap buffers.
            glfwSwapBuffers(g_window);
            // Poll window events.
            glfwPollEvents();
        }
    }

    // Cleanup graphical resources.
    cleanup();

    // Tear down OpenGL.
    tearDownOpenGL();
    /*system("pause");*/
    return isOk ? 0 : -1;
}